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
    spec = importlib.util.spec_from_file_location("console_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_model_and_tokenizer(config, device, ckpt_path=None):
    data_dir = ROOT / "data" / config.dataset
    meta_path = data_dir / "meta.pkl"

    if ckpt_path is None:
        ckpt_path = ROOT / config.out_dir / config.ckpt_name
    else:
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path

    tokenizer = load_tokenizer(meta_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
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

    return model, tokenizer


def print_help():
    print()
    print("Commands:")
    print("  :help              show help")
    print("  :exit              exit console")
    print("  :reset             clear prompt history")
    print("  :temp <value>      set temperature, e.g. :temp 0.8")
    print("  :topk <value>      set top_k, e.g. :topk 50")
    print("  :topk none         disable top_k")
    print("  :max <value>       set max_new_tokens, e.g. :max 200")
    print("  :show              show current generation settings")
    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_char.py",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path)

    device = config.device
    model, tokenizer = load_model_and_tokenizer(
        config,
        device,
        ckpt_path=args.ckpt,
    )

    temperature = args.temperature
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens

    history = ""

    print("MiniGPT Interactive Console")
    print("Type :help for commands.")
    print()

    while True:
        user_input = input("> ")

        if user_input.strip() == "":
            continue

        command = user_input.strip()

        if command == ":exit":
            break

        if command == ":help":
            print_help()
            continue

        if command == ":reset":
            history = ""
            print("context cleared")
            continue

        if command == ":show":
            print(f"temperature = {temperature}")
            print(f"top_k = {top_k}")
            print(f"max_new_tokens = {max_new_tokens}")
            print(f"context length = {len(history)} chars")
            continue

        if command.startswith(":temp "):
            value = command.split(maxsplit=1)[1]
            temperature = float(value)

            if temperature <= 0:
                raise ValueError("temperature must be positive")

            print(f"temperature = {temperature}")
            continue

        if command.startswith(":topk "):
            value = command.split(maxsplit=1)[1].strip()

            if value.lower() == "none":
                top_k = None
            else:
                top_k = int(value)

            print(f"top_k = {top_k}")
            continue

        if command.startswith(":max "):
            value = command.split(maxsplit=1)[1]
            max_new_tokens = int(value)

            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive")

            print(f"max_new_tokens = {max_new_tokens}")
            continue

        prompt = history + user_input

        ids = tokenizer.encode(prompt)
        context = torch.tensor([ids], dtype=torch.long, device=device)

        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        generated_text = tokenizer.decode(
            generated[0].detach().cpu().tolist()
        )

        new_text = generated_text[len(prompt):]

        print()
        print(new_text)
        print()

        history = generated_text


if __name__ == "__main__":
    main()