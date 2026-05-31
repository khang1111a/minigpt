import argparse
import sys
from pathlib import Path
import numpy as np



ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.tokenizers import CharTokenizer, ByteTokenizer


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

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if args.tokenizer == "char":
        tokenizer = CharTokenizer(text=text)
    elif args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {args.tokenizer}")

    ids = tokenizer.encode(text)

    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    ids = np.array(ids, dtype=dtype)

    n = int(len(ids) * args.train_ratio)

    train_ids = ids[:n]
    val_ids = ids[n:]

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    tokenizer.save(out_dir / "meta.pkl")

    print("text length:", len(text))
    print("total tokens:", len(ids))
    print("train tokens:", len(train_ids))
    print("val tokens:", len(val_ids))
    print("vocab size:", tokenizer.vocab_size)
    print("saved to:", out_dir)


if __name__ == "__main__":
    main()