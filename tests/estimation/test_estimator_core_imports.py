from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import unittest


class EstimatorCoreImportBoundaryTests(unittest.TestCase):
    def test_core_modules_import_without_loading_adapter_modules(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        script = """
import importlib
import sys

importlib.import_module("sim.estimation.core.ekf")
importlib.import_module("sim.estimation.core.eskf")
importlib.import_module("sim.estimation.stacks.layered_navigation")

loaded_adapters = sorted(
    name for name in sys.modules if name.startswith("sim.estimation.adapters")
)
if loaded_adapters:
    raise SystemExit(",".join(loaded_adapters))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"unexpected adapter imports: {result.stdout}{result.stderr}",
        )

    def test_legacy_monolithic_modules_are_removed(self) -> None:
        """Phase 8 gate: legacy eskf.py and replay.py must not exist."""
        estimation_dir = Path(__file__).resolve().parents[2] / "sim" / "estimation"
        self.assertFalse(
            (estimation_dir / "eskf.py").exists(),
            "Legacy eskf.py still present — phase 8 requires removal.",
        )
        self.assertFalse(
            (estimation_dir / "replay.py").exists(),
            "Legacy replay.py still present — phase 8 requires removal.",
        )

    def test_top_level_init_does_not_import_adapters(self) -> None:
        """Importing sim.estimation must not pull in adapter modules."""
        repository_root = Path(__file__).resolve().parents[2]
        script = """
import importlib
import sys

importlib.import_module("sim.estimation")

loaded_adapters = sorted(
    name for name in sys.modules if name.startswith("sim.estimation.adapters")
)
if loaded_adapters:
    raise SystemExit(",".join(loaded_adapters))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"top-level init loads adapter modules: {result.stdout}{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
