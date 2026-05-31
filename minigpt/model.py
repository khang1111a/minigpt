import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.dropout = nn.Dropout(dropout)

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
        wei = self.dropout(wei)
        # 用注意力权重 wei 对 v 做加权求和
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        # 拼接后的结果再做一次线性变换，让不同 head 的信息充分混合
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([
            head(x)
            for head in self.heads
        ], dim=-1)

        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        self.sa = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, n_layer, dropout):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 单头注意力模块
        #self.sa_head = Head(n_embd, n_embd, block_size)

        # 多头注意力模块
        #self.sa_head = MultiHeadAttention(n_embd, num_heads, block_size)
        
        # 前馈网络
        #self.ffwd = FeedForward(n_embd)

        # 归一化层
        #self.ln1 = nn.LayerNorm(n_embd)
        #self.ln2 = nn.LayerNorm(n_embd)

        # transformer block
        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos = torch.arange(T,device = idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb

        #x = x + self.sa_head(self.ln1(x))
        #x = x + self.ffwd(self.ln2(x))

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.reshape(B*T, C)
            targets_flat = targets.reshape(B*T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens,temperature = 1.0, top_k = None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))

                v, _ = torch.topk(logits, top_k)

                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim = -1)

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)

        return idx


