import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.dataset import get_batch
from minigpt.model import BigramLanguageModel
from minigpt.tokenizer import CharTokenizer

data_dir = ROOT / "data" / "tiny_text"

train_path = data_dir / "train.bin"
val_path = data_dir / "val.bin"
meta_path = data_dir / "meta.pkl"

train_data = np.fromfile(train_path, dtype=np.uint16)
val_data = np.fromfile(val_path, dtype=np.uint16)

tokenizer = CharTokenizer.load(meta_path)

print("train data shape:", train_data.shape)
print("val data shape:", val_data.shape)
print("vocab size:", tokenizer.vocab_size)

batch_size = 4
block_size = 8

n_embd = 32

model = BigramLanguageModel(tokenizer.vocab_size,n_embd)

x, y = get_batch(
    split="train",
    train_data=train_data,
    val_data=val_data,
    batch_size=batch_size,
    block_size=block_size,
)

logits, loss = model(x, y)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("logits shape:", logits.shape)
print("loss:", loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

max_iters = 200

for step in range(max_iters):
    x, y = get_batch(
        split="train",
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        block_size=block_size,
    )

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step}: loss {loss.item():.4f}")


context = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(context, max_new_tokens=100)

generated_text = tokenizer.decode(generated[0].tolist())

print("generated text:")
print(generated_text)

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

ckpt_path = output_dir / "bigram.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
    },
    ckpt_path,
)

print("saved checkpoint:", ckpt_path)
