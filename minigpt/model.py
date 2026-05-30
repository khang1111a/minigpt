import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)

        # scale
        head_size = k.shape[-1]
        wei = wei * head_size ** -0.5

        # mask
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )
        
        wei = F.softmax(wei, dim=-1)

        # 用注意力权重 wei 对 v 做加权求和
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ])
        # 拼接后的结果再做一次线性变换，让不同 head 的信息充分混合
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([
            head(x)
            for head in self.heads
        ], dim=-1)

        out = self.proj(out)

        return out



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 单头注意力模块
        #self.sa_head = Head(n_embd, n_embd, block_size)

        # 多头注意力模块
        self.sa_head = MultiHeadAttention(n_embd, num_heads, block_size)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos = torch.arange(T,device = idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb

        x = self.sa_head(x)

        logits = self.lm_head(x)

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
            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)

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

    n_embd = 32
    num_heads = 4

    model = BigramLanguageModel(vocab_size,n_embd,block_size,num_heads)

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
