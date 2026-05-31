import subprocess
import sys


def test_sample_help_includes_ckpt_option():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sample.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--ckpt" in result.stdout