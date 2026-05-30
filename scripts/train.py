import sys
from pathlib import Path
import importlib.util
import numpy as np
import torch

import csv


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.dataset import get_batch
from minigpt.model import GPTLanguageModel
from minigpt.tokenizer import CharTokenizer

config_path = ROOT / "configs" / "train_char.py"

spec = importlib.util.spec_from_file_location("train_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

data_dir = ROOT / "data" / config.dataset

train_path = data_dir / "train.bin"
val_path = data_dir / "val.bin"
meta_path = data_dir / "meta.pkl"

train_data = np.fromfile(train_path, dtype=np.uint16)
val_data = np.fromfile(val_path, dtype=np.uint16)

tokenizer = CharTokenizer.load(meta_path)

model = GPTLanguageModel(
    tokenizer.vocab_size,
    config.n_embd,
    config.block_size,
    config.num_heads,
    config.n_layer,
    config.dropout,
)

model.to(config.device)

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
        loss_value = loss.item()
        loss_history.append((step, loss_value))

        print(f"step {step}: loss {loss_value:.4f}")

results_dir = ROOT / config.results_dir
results_dir.mkdir(exist_ok=True)

loss_path = results_dir / config.loss_name

with open(loss_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["step", "loss"])

    for step, loss_value in loss_history:
        writer.writerow([step, loss_value])

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