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

def test_eval_help_includes_expected_options():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--ckpt" in result.stdout
    assert "--eval-iters" in result.stdout
    assert "--split" in result.stdout

def test_console_help_includes_ckpt_option():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/console.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--ckpt" in result.stdout