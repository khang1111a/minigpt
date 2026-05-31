import sys
from pathlib import Path
import importlib.util
import numpy as np
import torch

import csv
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.dataset import get_batch
from minigpt.model import GPTLanguageModel
from minigpt.tokenizers import load_tokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    default="configs/train_char.py",
)

args = parser.parse_args()

config_path = Path(args.config)

if not config_path.is_absolute():
    config_path = ROOT / config_path

spec = importlib.util.spec_from_file_location("train_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

seed = getattr(config, "seed", 1337)

torch.manual_seed(seed)
np.random.seed(seed)

seed = getattr(config, "seed", 1337)

torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data_dir = ROOT / "data" / config.dataset

train_path = data_dir / "train.bin"
val_path = data_dir / "val.bin"
meta_path = data_dir / "meta.pkl"

tokenizer = load_tokenizer(meta_path)

data_dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32

train_data = np.fromfile(train_path, dtype=data_dtype)
val_data = np.fromfile(val_path, dtype=data_dtype)

model = GPTLanguageModel(
    tokenizer.vocab_size,
    config.n_embd,
    config.block_size,
    config.num_heads,
    config.n_layer,
    config.dropout,
)

model.to(config.device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    model.eval()

    out = {}

    eval_iters = getattr(config, "eval_iters", 10)

    for split in ["train", "val"]:
        data = train_data if split == "train" else val_data

        if len(data) <= config.block_size:
            out[split] = None
            continue

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

            logits, loss = model(x, y)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
)

loss_history = []
for step in range(config.max_iters):
    x, y = get_batch(
        split="train",
        train_data=train_data,
        val_data=val_data,
        batch_size=config.batch_size,
        block_size=config.block_size,
    )

    x = x.to(config.device)
    y = y.to(config.device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % config.eval_interval == 0:
        losses = estimate_loss(
            model=model,
            train_data=train_data,
            val_data=val_data,
            config=config,
        )

        train_loss = losses["train"]
        val_loss = losses["val"]

        loss_history.append((step, train_loss, val_loss))

        if val_loss is None:
            print(f"step {step}: train loss {train_loss:.4f}, val loss N/A")
        else:
            print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

results_dir = ROOT / config.results_dir
results_dir.mkdir(exist_ok=True)

loss_path = results_dir / config.loss_name

with open(loss_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["step", "train_loss", "val_loss"])

    for step, train_loss, val_loss in loss_history:
        writer.writerow([
            step,
            train_loss,
            "" if val_loss is None else val_loss,
        ])

print("saved loss history:", loss_path)


output_dir = ROOT / config.out_dir
output_dir.mkdir(exist_ok=True)

ckpt_path = output_dir / config.ckpt_name

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "num_heads": config.num_heads,
        "n_layer": config.n_layer,
        "dropout": config.dropout,
        "seed": seed,
    },
    ckpt_path,
)

print("saved checkpoint:", ckpt_path)


if config.generate_after_train:
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    generated = model.generate(
        context,
        max_new_tokens=config.max_new_tokens,
    )

    generated_text = tokenizer.decode(generated[0].tolist())

    print("generated text:")
    print(generated_text)