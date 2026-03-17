from __future__ import annotations

import io
import json
import unittest

import pandas as pd
from pymavlink.dialects.v20 import common as mavlink2

from sim.sitl.adapters import JsonUdpAdapter, MavlinkCommonAdapter
from sim.sitl.replay import ReplayClock


class ReplayClockTests(unittest.TestCase):
    def test_sync_to_time_chooses_first_sample_at_or_after_target(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 0.01, 0.02, 0.03],
                "accelerometer_x": [0.0, 1.0, 2.0, 3.0],
            }
        )
        clock = ReplayClock(frame)

        clock.sync_to_time(0.019)

        self.assertEqual(clock.index, 2)
        self.assertAlmostEqual(clock.current_time_s(), 0.02)

    def test_step_clamps_at_end(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 0.01, 0.02],
                "accelerometer_x": [0.0, 1.0, 2.0],
            }
        )
        clock = ReplayClock(frame)

        clock.step(10)

        self.assertEqual(clock.index, 2)
        self.assertTrue(clock.at_end)


class JsonAdapterTests(unittest.TestCase):
    def test_json_adapter_normalizes_nan_to_null(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0],
                "barometer_v1": [float("nan")],
                "accelerometer_x": [1.5],
            }
        )
        clock = ReplayClock(frame)
        sample = clock.current_sample()

        packet = JsonUdpAdapter().encode_event(event="snapshot", clock=clock, sample=sample)[0]
        payload = json.loads(packet.payload.decode("utf-8"))

        self.assertEqual(payload["event"], "snapshot")
        self.assertIsNone(payload["sample"]["row"]["barometer_v1"])
        self.assertEqual(payload["sample"]["row"]["accelerometer_x"], 1.5)


class MavlinkCommonAdapterTests(unittest.TestCase):
    def test_adapter_emits_standard_common_messages(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0],
                "accelerometer_x": [0.1],
                "accelerometer_y": [0.2],
                "accelerometer_z": [-9.7],
                "gyroscope_x": [0.01],
                "gyroscope_y": [0.02],
                "gyroscope_z": [0.03],
                "barometer_v1": [96453.7],
                "gnss_x": [33.4986538],
                "gnss_y": [-99.3375871],
                "gnss_z": [413.9],
            }
        )
        clock = ReplayClock(frame)
        sample = clock.current_sample()

        packets = MavlinkCommonAdapter(frame).encode_event(
            event="snapshot",
            clock=clock,
            sample=sample,
        )
        messages = _decode_mavlink_packets([packet.payload for packet in packets])

        self.assertEqual(
            [message.get_type() for message in messages],
            ["SYSTEM_TIME", "HIL_SENSOR", "HIL_GPS"],
        )

        system_time = messages[0]
        hil_sensor = messages[1]
        hil_gps = messages[2]

        self.assertEqual(system_time.time_boot_ms, 0)
        self.assertAlmostEqual(hil_sensor.xacc, 0.1, places=6)
        self.assertAlmostEqual(hil_sensor.zgyro, 0.03, places=6)
        self.assertEqual(hil_gps.fix_type, 3)
        self.assertEqual(hil_gps.lat, int(round(33.4986538 * 1e7)))


def _decode_mavlink_packets(payloads: list[bytes]) -> list:
    parser = mavlink2.MAVLink(io.BytesIO())
    messages = []
    for payload in payloads:
        for byte in payload:
            message = parser.parse_char(bytes([byte]))
            if message is not None:
                messages.append(message)
    return messages


if __name__ == "__main__":
    unittest.main()