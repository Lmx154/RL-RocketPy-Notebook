"""UDP control and output bridge for layered SITL replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import JsonUdpAdapter, MavlinkCommonAdapter, ProtocolAdapter
from .replay import ReplayClock, ReplaySample, load_replay_telemetry


@dataclass(slots=True)
class UdpEndpoint:
    """Remote UDP endpoint."""

    host: str
    port: int

    def as_tuple(self) -> tuple[str, int]:
        return (self.host, self.port)


@dataclass(slots=True)
class UdpOutputChannel:
    """Output channel pairing a protocol adapter with a UDP endpoint."""

    adapter: ProtocolAdapter
    endpoint: UdpEndpoint


class UdpDatagramTransport:
    """Minimal UDP sender transport used by replay output channels."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def sendto(self, payload: bytes, endpoint: UdpEndpoint) -> None:
        self._socket.sendto(payload, endpoint.as_tuple())

    def close(self) -> None:
        self._socket.close()


class CsvReplayUdpService:
    """Replay CSV telemetry through layered UDP outputs with UDP control."""

    def __init__(
        self,
        telemetry,
        *,
        telemetry_path: Path | None = None,
        time_column: str = "time_s",
        output_channels: list[UdpOutputChannel] | None = None,
        control_host: str = "0.0.0.0",
        control_port: int = 14600,
    ) -> None:
        self.telemetry_path = telemetry_path
        self.clock = ReplayClock(telemetry, time_column=time_column)
        self.control_host = control_host
        self.control_port = int(control_port)
        self.output_channels = list(output_channels or [])
        self._sender = UdpDatagramTransport()
        self._control_transport: asyncio.DatagramTransport | None = None
        self._playback_task: asyncio.Task[Any] | None = None
        self._state_lock = asyncio.Lock()

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UdpControlProtocol(self),
            local_addr=(self.control_host, self.control_port),
        )
        self._control_transport = transport
        try:
            print(
                f"SITL UDP replay control on udp://{self.control_host}:{self.control_port} "
                f"with {len(self.output_channels)} output channel(s)"
            )
            await asyncio.Future()
        finally:
            if self._playback_task is not None:
                self._playback_task.cancel()
            transport.close()
            self._sender.close()

    async def handle_control_datagram(
        self,
        payload: bytes,
        address: tuple[str, int],
        transport: asyncio.DatagramTransport,
    ) -> None:
        try:
            command = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            transport.sendto(
                json.dumps({"type": "error", "message": "Invalid JSON command"}).encode("utf-8"),
                address,
            )
            return

        response = await self.apply_control_command(command)
        transport.sendto(json.dumps(response).encode("utf-8"), address)

    async def apply_control_command(self, command: dict[str, Any]) -> dict[str, Any]:
        op = str(command.get("op", "")).strip().lower()
        if not op:
            return {"type": "error", "message": "Missing 'op' field"}

        if op == "status":
            return self._ack(op)

        if op == "reset":
            async with self._state_lock:
                self.clock.reset()
            await self._emit_current_sample(event="reset")
            return self._ack(op)

        if op == "pause":
            await self._set_playing(False)
            return self._ack(op)

        if op == "play":
            rate = max(float(command.get("rate", 1.0)), 1e-6)
            async with self._state_lock:
                self.clock.replay_rate = rate
            await self._set_playing(True)
            return self._ack(op)

        if op == "sync":
            if "time_s" not in command:
                return {"type": "error", "message": "'sync' requires 'time_s'"}
            async with self._state_lock:
                self.clock.sync_to_time(float(command["time_s"]))
            await self._emit_current_sample(event="sync")
            return self._ack(op)

        if op == "seek_index":
            if "index" not in command:
                return {"type": "error", "message": "'seek_index' requires 'index'"}
            async with self._state_lock:
                self.clock.seek_index(int(command["index"]))
            await self._emit_current_sample(event="seek_index")
            return self._ack(op)

        if op == "step":
            count = max(int(command.get("count", 1)), 0)
            async with self._state_lock:
                self.clock.step(count)
            await self._emit_current_sample(event="step")
            return self._ack(op)

        return {"type": "error", "message": f"Unknown op '{op}'"}

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

                await self._emit_current_sample(event="tick")

            await self._emit_current_sample(event="replay_finished")
        except asyncio.CancelledError:
            return

    async def _emit_current_sample(self, event: str) -> None:
        sample = self.clock.current_sample()
        for channel in self.output_channels:
            for packet in channel.adapter.encode_event(event=event, clock=self.clock, sample=sample):
                self._sender.sendto(packet.payload, channel.endpoint)

    def _ack(self, op: str) -> dict[str, Any]:
        return {
            "type": "ack",
            "op": op,
            "state": self.clock.snapshot(),
        }


class _UdpControlProtocol(asyncio.DatagramProtocol):
    def __init__(self, service: CsvReplayUdpService) -> None:
        self._service = service
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        if self._transport is None:
            return
        asyncio.create_task(self._service.handle_control_datagram(data, address, self._transport))


def build_output_channels(
    telemetry,
    *,
    target_host: str,
    json_target_port: int | None,
    mavlink_target_port: int | None,
    mavlink_system_id: int,
    mavlink_component_id: int,
) -> list[UdpOutputChannel]:
    channels: list[UdpOutputChannel] = []

    if json_target_port is not None:
        channels.append(
            UdpOutputChannel(
                adapter=JsonUdpAdapter(),
                endpoint=UdpEndpoint(target_host, int(json_target_port)),
            )
        )

    if mavlink_target_port is not None:
        channels.append(
            UdpOutputChannel(
                adapter=MavlinkCommonAdapter(
                    telemetry,
                    system_id=mavlink_system_id,
                    component_id=mavlink_component_id,
                ),
                endpoint=UdpEndpoint(target_host, int(mavlink_target_port)),
            )
        )

    return channels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay CSV telemetry over layered UDP outputs for SITL",
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
    parser.add_argument(
        "--control-host",
        type=str,
        default="0.0.0.0",
        help="UDP host for replay control commands",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=14600,
        help="UDP port for replay control commands",
    )
    parser.add_argument(
        "--target-host",
        type=str,
        default="127.0.0.1",
        help="Remote UDP host for replay output adapters",
    )
    parser.add_argument(
        "--json-target-port",
        type=int,
        default=None,
        help="Remote UDP port for debug JSON replay datagrams",
    )
    parser.add_argument(
        "--mavlink-target-port",
        type=int,
        default=14550,
        help="Remote UDP port for MAVLink common.xml datagrams",
    )
    parser.add_argument(
        "--mavlink-system-id",
        type=int,
        default=1,
        help="MAVLink system id",
    )
    parser.add_argument(
        "--mavlink-component-id",
        type=int,
        default=1,
        help="MAVLink component id",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="time_s",
        help="Telemetry time column used for replay clock",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    telemetry, telemetry_path = load_replay_telemetry(
        args.telemetry,
        logs_directory=args.logs_directory,
        time_column=args.time_column,
    )
    channels = build_output_channels(
        telemetry,
        target_host=args.target_host,
        json_target_port=args.json_target_port,
        mavlink_target_port=args.mavlink_target_port,
        mavlink_system_id=args.mavlink_system_id,
        mavlink_component_id=args.mavlink_component_id,
    )
    service = CsvReplayUdpService(
        telemetry,
        telemetry_path=telemetry_path,
        time_column=args.time_column,
        output_channels=channels,
        control_host=args.control_host,
        control_port=args.control_port,
    )
    await service.run_forever()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()