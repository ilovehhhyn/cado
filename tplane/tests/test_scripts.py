"""The smoke script refuses to run unless explicitly opted in, and rejects any opt-in value other than 1."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "della_smoke.py"


def run_script(value: str | None, run_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ["PATH"], "LANG": "C"}
    if value is not None:
        env["TPLANE_DEVICE_TESTS"] = value
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_smoke_script_refuses_without_opt_in(tmp_path: Path) -> None:
    completed = run_script(None, tmp_path / "run")

    assert completed.returncode == 2
    assert "TPLANE_DEVICE_TESTS must be set to 1 to run the gpu smoke test" in completed.stderr
    assert not (tmp_path / "run").exists()


def test_smoke_script_rejects_an_unknown_opt_in_value(tmp_path: Path) -> None:
    completed = run_script("yes", tmp_path / "run")

    assert completed.returncode == 2
    assert "TPLANE_DEVICE_TESTS must be 1 or unset, got 'yes'" in completed.stderr
    assert not (tmp_path / "run").exists()
