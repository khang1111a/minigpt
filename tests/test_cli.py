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
    assert "--out" in result.stdout
    assert "--csv" in result.stdout

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
    assert "--log" in result.stdout

def test_train_help_includes_run_options():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-id" in result.stdout
    assert "--runs-dir" in result.stdout
