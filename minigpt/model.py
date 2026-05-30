import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.reshape(B*T, C)
            targets_flat = targets.reshape(B*T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim = -1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)

        return idx


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import numpy as np

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))

    from minigpt.dataset import get_batch
    from minigpt.tokenizer import CharTokenizer

    data_dir = ROOT / "data" / "tiny_text"

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    meta_path = data_dir / "meta.pkl"

    train_data = np.fromfile(train_path, dtype=np.uint16)
    val_data = np.fromfile(val_path, dtype=np.uint16)

    tokenizer = CharTokenizer.load(meta_path)

    vocab_size = tokenizer.vocab_size
    batch_size = 4
    block_size = 8

    x, y = get_batch(
        split="train",
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        block_size=block_size,
    )

    model = BigramLanguageModel(vocab_size)

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

    generated = model.generate(context, max_new_tokens=10)

    print("generated shape:", generated.shape)
    print("generated ids:", generated)
    print("generated text:", tokenizer.decode(generated[0].tolist()))