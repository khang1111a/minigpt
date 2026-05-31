import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.tokenizers import build_tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="char")
    parser.add_argument("--train-ratio", type=float, default=0.9)

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    if not 0 < args.train_ratio < 1:
        raise ValueError(f"train-ratio must be between 0 and 1, got {args.train_ratio}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = build_tokenizer(args.tokenizer, text=text)

    ids = tokenizer.encode(text)

    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    ids = np.array(ids, dtype=dtype)

    n = int(len(ids) * args.train_ratio)

    train_ids = ids[:n]
    val_ids = ids[n:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    meta_path = out_dir / "meta.pkl"

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    tokenizer.save(meta_path)

    print("input path:", input_path)
    print("output dir:", out_dir)
    print("tokenizer:", args.tokenizer)
    print("dtype:", dtype)
    print("text length:", len(text))
    print("total tokens:", len(ids))
    print("train tokens:", len(train_ids))
    print("val tokens:", len(val_ids))
    print("vocab size:", tokenizer.vocab_size)
    print("saved:", train_path)
    print("saved:", val_path)
    print("saved:", meta_path)


if __name__ == "__main__":
    main()