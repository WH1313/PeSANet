import subprocess
import sys
from pathlib import Path


def test_smoke_forward_script_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "examples/smoke_forward.py",
            "--grid-size",
            "64",
            "--steps",
            "1",
            "--modes",
            "4",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "sequence_shape=[1, 2, 2, 64, 64]" in result.stdout
