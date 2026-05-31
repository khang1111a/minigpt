import subprocess
import sys
from pathlib import Path

import numpy as np

from minigpt.tokenizers import load_tokenizer


def test_prepare_data_char_outputs_expected_files(tmp_path):
    input_path = tmp_path / "input.txt"
    output_dir = tmp_path / "tiny_text_char"

    input_path.write_text("hello world\nhello minigpt", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--input",
            str(input_path),
            "--out",
            str(output_dir),
            "--tokenizer",
            "char",
            "--train-ratio",
            "0.8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.pkl"

    assert train_path.exists()
    assert val_path.exists()
    assert meta_path.exists()

    tokenizer = load_tokenizer(meta_path)
    assert tokenizer.vocab_size == len(set("hello world\nhello minigpt"))

    train_ids = np.fromfile(train_path, dtype=np.uint16)
    val_ids = np.fromfile(val_path, dtype=np.uint16)

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert "tokenizer: char" in result.stdout