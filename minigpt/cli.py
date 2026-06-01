import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


COMMAND_TO_SCRIPT = {
    "prepare": "scripts/prepare_data.py",
    "train": "scripts/train.py",
    "eval": "scripts/eval.py",
    "sample": "scripts/sample.py",
    "console": "scripts/console.py",
    "plot": "scripts/plot_loss.py",
}


def run_script(command, args):
    script = ROOT / COMMAND_TO_SCRIPT[command]
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        cwd = ROOT,
    )
    return completed.returncode


def main(argv = None):
    parser = argparse.ArgumentParser(
        prog = "minigpt",
        description = "MiniGPT command line interface.",
    )

    parser.add_argument(
        "command",
        choices = sorted(COMMAND_TO_SCRIPT),
        help = "Command to run.",
    )

    parser.add_argument(
        "args",
        nargs = argparse.REMAINDER,
        help = "Arguments passed to the selected command.",
    )

    parsed = parser.parse_args(argv)
    raise SystemExit(run_script(parsed.command, parsed.args))