import subprocess
import sys

import pytest


def run_minigpt(*args):
    return subprocess.run(
        [sys.executable, "-m", "minigpt", *args],
        check = True,
        capture_output = True,
        text = True,
    )


def test_minigpt_help_includes_commands():
    result = run_minigpt("--help")

    assert "MiniGPT command line interface." in result.stdout

    for command in ["prepare", "train", "eval", "sample", "console", "plot"]:
        assert command in result.stdout


@pytest.mark.parametrize(
    ("command", "expected_flags"),
    [
        ("prepare", ["--input", "--out", "--tokenizer"]),
        ("train", ["--config", "--resume"]),
        ("eval", ["--config", "--ckpt", "--eval-iters", "--split"]),
        ("sample", ["--config", "--prompt", "--ckpt", "--temperature", "--top-k"]),
        ("console", ["--config", "--ckpt", "--log"]),
        ("plot", ["--loss", "--out"]),
    ],
)
def test_minigpt_forwards_subcommand_help(command, expected_flags):
    result = run_minigpt(command, "--help")

    for flag in expected_flags:
        assert flag in result.stdout