"""WebSocket bridge that replays CSV telemetry for firmware SITL integration."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve

from sim.estimation.adapters.rocketpy_replay import find_latest_telemetry_log


@dataclass(slots=True)
class ReplayClock:
    """Mutable replay cursor and time synchronization helper."""

    telemetry: pd.DataFrame
    time_column: str = "time_s"
    index: int = 0
    playing: bool = False
    replay_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.time_column not in self.telemetry.columns:
            raise ValueError(f"Missing time column '{self.time_column}' in telemetry frame")
        if self.telemetry.empty:
            raise ValueError("Telemetry frame is empty")
        self.telemetry = self.telemetry.sort_values(self.time_column).reset_index(drop=True)

    @property
    def times_s(self) -> np.ndarray:
        return self.telemetry[self.time_column].to_numpy(dtype=float)

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.telemetry) - 1

    def clamp_index(self, index: int) -> int:
        return int(max(0, min(index, len(self.telemetry) - 1)))

    def reset(self) -> None:
        self.index = 0

    def sync_to_time(self, target_time_s: float) -> int:
        position = int(np.searchsorted(self.times_s, float(target_time_s), side="left"))
        self.index = self.clamp_index(position)
        return self.index

    def step(self, count: int = 1) -> int:
        self.index = self.clamp_index(self.index + max(int(count), 0))
        return self.index

    def seek_index(self, index: int) -> int:
        self.index = self.clamp_index(index)
        return self.index

    def current_row(self) -> dict[str, Any]:
        row = self.telemetry.iloc[self.index].to_dict()
        return {column: _json_scalar(value) for column, value in row.items()}

    def current_time_s(self) -> float:
        return float(self.times_s[self.index])

    def dt_to_next_s(self) -> float:
        if self.at_end:
            return 0.0
        current = self.current_time_s()
        nxt = float(self.times_s[self.index + 1])
        return max(0.0, nxt - current)

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time_s": self.current_time_s(),
            "at_end": self.at_end,
            "playing": self.playing,
            "replay_rate": self.replay_rate,
            "total_samples": int(len(self.telemetry)),
        }


def load_replay_telemetry(
    telemetry: str | Path | pd.DataFrame | None,
    *,
    logs_directory: str | Path,
    time_column: str = "time_s",
) -> tuple[pd.DataFrame, Path | None]:
    """Load replay telemetry from DataFrame, explicit path, or latest logs CSV."""

    telemetry_path: Path | None = None
    if telemetry is None:
        telemetry_path = find_latest_telemetry_log(logs_directory)
        frame = pd.read_csv(telemetry_path)
    elif isinstance(telemetry, pd.DataFrame):
        frame = telemetry.copy()
    else:
        telemetry_path = Path(telemetry)
        frame = pd.read_csv(telemetry_path)

    if time_column not in frame.columns:
        raise ValueError(f"Telemetry is missing required '{time_column}' column")

    frame = frame.sort_values(time_column).reset_index(drop=True)
    return frame, telemetry_path


class SitlCsvReplayServer:
    """Replay server with `/clock` and `/sensors` WebSocket endpoints."""

    def __init__(
        self,
        telemetry: str | Path | pd.DataFrame | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        logs_directory: str | Path = Path(__file__).resolve().parents[2] / "logs",
        time_column: str = "time_s",
    ) -> None:
        frame, telemetry_path = load_replay_telemetry(
            telemetry,
            logs_directory=logs_directory,
            time_column=time_column,
        )
        self.host = host
        self.port = int(port)
        self.telemetry_path = telemetry_path
        self.clock = ReplayClock(frame, time_column=time_column)
        self._state_lock = asyncio.Lock()
        self._clock_clients: set[WebSocketServerProtocol] = set()
        self._sensor_clients: set[WebSocketServerProtocol] = set()
        self._playback_task: asyncio.Task[Any] | None = None

    async def run_forever(self) -> None:
        async with serve(self._route_connection, self.host, self.port):
            print(
                f"SITL replay server on ws://{self.host}:{self.port} "
                "(clock endpoint: /clock, sensors endpoint: /sensors)"
            )
            await asyncio.Future()

    async def _route_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        if path == "/clock":
            await self._clock_session(websocket)
            return
        if path == "/sensors":
            await self._sensor_session(websocket)
            return

        await websocket.send(
            json.dumps(
                {
                    "type": "error",
                    "message": f"Unknown endpoint '{path}'. Use /clock or /sensors.",
                }
            )
        )
        await websocket.close(code=1008, reason="Unknown endpoint")

    async def _clock_session(self, websocket: WebSocketServerProtocol) -> None:
        self._clock_clients.add(websocket)
        try:
            await websocket.send(json.dumps(self._clock_message("status")))
            async for raw_message in websocket:
                await self._handle_clock_command(websocket, raw_message)
        except ConnectionClosed:
            return
        finally:
            self._clock_clients.discard(websocket)

    async def _sensor_session(self, websocket: WebSocketServerProtocol) -> None:
        self._sensor_clients.add(websocket)
        try:
            await websocket.send(json.dumps(self._sensor_message(event="snapshot")))
            async for _ in websocket:
                # The sensors endpoint is push-based; incoming messages are ignored.
                pass
        except ConnectionClosed:
            return
        finally:
            self._sensor_clients.discard(websocket)

    async def _handle_clock_command(self, websocket: WebSocketServerProtocol, raw_message: str) -> None:
        try:
            command = json.loads(raw_message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON command"}))
            return

        op = str(command.get("op", "")).strip().lower()
        if not op:
            await websocket.send(json.dumps({"type": "error", "message": "Missing 'op' field"}))
            return

        if op == "status":
            await websocket.send(json.dumps(self._clock_message("status")))
            return

        if op == "reset":
            async with self._state_lock:
                self.clock.reset()
            await self._broadcast_tick(event="reset")
            return

        if op == "pause":
            await self._set_playing(False)
            await self._broadcast_clock(event="paused")
            return

        if op == "play":
            rate = max(float(command.get("rate", 1.0)), 1e-6)
            async with self._state_lock:
                self.clock.replay_rate = rate
            await self._set_playing(True)
            await self._broadcast_clock(event="playing")
            return

        if op == "sync":
            if "time_s" not in command:
                await websocket.send(
                    json.dumps({"type": "error", "message": "'sync' requires 'time_s'"})
                )
                return
            async with self._state_lock:
                self.clock.sync_to_time(float(command["time_s"]))
            await self._broadcast_tick(event="sync")
            return

        if op == "seek_index":
            if "index" not in command:
                await websocket.send(
                    json.dumps({"type": "error", "message": "'seek_index' requires 'index'"})
                )
                return
            async with self._state_lock:
                self.clock.seek_index(int(command["index"]))
            await self._broadcast_tick(event="seek_index")
            return

        if op == "step":
            count = max(int(command.get("count", 1)), 0)
            async with self._state_lock:
                self.clock.step(count)
            await self._broadcast_tick(event="step")
            return

        await websocket.send(json.dumps({"type": "error", "message": f"Unknown op '{op}'"}))

    async def _set_playing(self, playing: bool) -> None:
        async with self._state_lock:
            self.clock.playing = playing

        if not playing:
            if self._playback_task is not None:
                self._playback_task.cancel()
                self._playback_task = None
            return

        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_loop())

    async def _playback_loop(self) -> None:
        try:
            while True:
                async with self._state_lock:
                    if not self.clock.playing:
                        return
                    if self.clock.at_end:
                        self.clock.playing = False
                        break

                    dt_s = self.clock.dt_to_next_s()
                    replay_rate = self.clock.replay_rate

                await asyncio.sleep(dt_s / replay_rate if dt_s > 0.0 else 0.0)

                async with self._state_lock:
                    if not self.clock.playing:
                        return
                    self.clock.step(1)

                await self._broadcast_tick(event="tick")

            await self._broadcast_clock(event="replay_finished")
        except asyncio.CancelledError:
            return

    async def _broadcast_tick(self, event: str) -> None:
        await self._broadcast_clock(event=event)
        await self._broadcast_sensors(event=event)

    async def _broadcast_clock(self, event: str) -> None:
        message = json.dumps(self._clock_message(event))
        await self._broadcast(self._clock_clients, message)

    async def _broadcast_sensors(self, event: str) -> None:
        message = json.dumps(self._sensor_message(event))
        await self._broadcast(self._sensor_clients, message)

    async def _broadcast(self, clients: set[WebSocketServerProtocol], message: str) -> None:
        stale_clients: list[WebSocketServerProtocol] = []
        for client in clients:
            try:
                await client.send(message)
            except ConnectionClosed:
                stale_clients.append(client)
        for stale in stale_clients:
            clients.discard(stale)

    def _clock_message(self, event: str) -> dict[str, Any]:
        return {
            "type": "clock",
            "event": event,
            "source": "csv_replay",
            "state": self.clock.snapshot(),
        }

    def _sensor_message(self, event: str) -> dict[str, Any]:
        return {
            "type": "sensors",
            "event": event,
            "source": "csv_replay",
            "time_s": self.clock.current_time_s(),
            "row": self.clock.current_row(),
        }


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay CSV telemetry over WebSockets for firmware SITL",
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default=None,
        help="Path to virtual_sensors_full_rate CSV (defaults to latest log)",
    )
    parser.add_argument(
        "--logs-directory",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "logs"),
        help="Directory searched when --telemetry is omitted",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--time-column",
        type=str,
        default="time_s",
        help="Telemetry time column used for replay clock",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    server = SitlCsvReplayServer(
        telemetry=args.telemetry,
        logs_directory=args.logs_directory,
        host=args.host,
        port=args.port,
        time_column=args.time_column,
    )
    await server.run_forever()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
