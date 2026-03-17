from __future__ import annotations

import unittest

import pandas as pd

from sim.sitl.websocket_bridge import ReplayClock


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

    def test_current_row_maps_nan_to_none(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0],
                "barometer_v1": [float("nan")],
                "accelerometer_x": [1.5],
            }
        )
        clock = ReplayClock(frame)

        row = clock.current_row()

        self.assertIsNone(row["barometer_v1"])
        self.assertEqual(row["accelerometer_x"], 1.5)


if __name__ == "__main__":
    unittest.main()
