import argparse
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.model import GPTLanguageModel
from minigpt.tokenizers import load_tokenizer


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("sample_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_char.py",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path)

    device = config.device

    data_dir = ROOT / "data" / config.dataset
    meta_path = data_dir / "meta.pkl"

    ckpt_path = ROOT / config.out_dir / config.ckpt_name

    tokenizer = load_tokenizer(meta_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
    )

    seed = args.seed

    if seed is None:
        seed = checkpoint.get("seed", getattr(config, "seed", 1337))

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
                f"checkpoint missing key: {key}. "
                f"Please rerun scripts/train.py to save a new GPT checkpoint."
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
    model.to(device)
    model.eval()

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else config.max_new_tokens
    )

    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt)
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(
        context,
        max_new_tokens = max_new_tokens,
        temperature = args.temperature,
        top_k = args.top_k,
    )

    generated_text = tokenizer.decode(
        generated[0].detach().cpu().tolist()
    )

    print(generated_text)


if __name__ == "__main__":
    main()