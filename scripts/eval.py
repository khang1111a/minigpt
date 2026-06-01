import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.dataset import get_batch
from minigpt.model import GPTLanguageModel
from minigpt.tokenizers import load_tokenizer


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("eval_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


@torch.no_grad()
def estimate_split_loss(model, split, train_data, val_data, config, eval_iters):
    model.eval()

    data = train_data if split == "train" else val_data

    if len(data) <= config.block_size:
        return None

    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        x, y = get_batch(
            split=split,
            train_data=train_data,
            val_data=val_data,
            batch_size=config.batch_size,
            block_size=config.block_size,
        )

        x = x.to(config.device)
        y = y.to(config.device)

        _, loss = model(x, y)
        losses[k] = loss.item()

    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--eval-iters",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    ckpt_path = Path(args.ckpt)

    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    config = load_config(config_path)

    eval_iters = (
        args.eval_iters
        if args.eval_iters is not None
        else getattr(config, "eval_iters", 10)
    )

    data_dir = ROOT / "data" / config.dataset
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    meta_path = data_dir / "meta.pkl"

    tokenizer = load_tokenizer(meta_path)

    data_dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32

    train_data = np.fromfile(train_path, dtype=data_dtype)
    val_data = np.fromfile(val_path, dtype=data_dtype)

    checkpoint = torch.load(
        ckpt_path,
        map_location=config.device,
    )

    required_keys = [
        "model_state_dict",
        "vocab_size",
        "block_size",
        "n_embd",
        "num_heads",
        "n_layer",
        "dropout",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"checkpoint missing key: {key}"
            )

    if tokenizer.vocab_size != checkpoint["vocab_size"]:
        raise ValueError(
            f"vocab size mismatch: tokenizer={tokenizer.vocab_size}, "
            f"checkpoint={checkpoint['vocab_size']}"
        )

    model = GPTLanguageModel(
        checkpoint["vocab_size"],
        checkpoint["n_embd"],
        checkpoint["block_size"],
        checkpoint["num_heads"],
        checkpoint["n_layer"],
        checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    print(f"checkpoint: {ckpt_path}")
    print(f"dataset: {config.dataset}")
    print(f"device: {config.device}")
    print(f"eval_iters: {eval_iters}")

    for split in splits:
        loss = estimate_split_loss(
            model=model,
            split=split,
            train_data=train_data,
            val_data=val_data,
            config=config,
            eval_iters=eval_iters,
        )

        if loss is None:
            print(f"{split}_loss: N/A")
            print(f"{split}_perplexity: N/A")
            continue

        perplexity = math.exp(loss)

        print(f"{split}_loss: {loss:.4f}")
        print(f"{split}_perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()