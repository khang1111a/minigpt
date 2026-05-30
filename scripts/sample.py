import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.model import GPTLanguageModel
from minigpt.tokenizer import CharTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = ROOT / "data" / "tiny_text"
output_dir = ROOT / "outputs"

meta_path = data_dir / "meta.pkl"
ckpt_path = output_dir / "gpt.pt"

tokenizer = CharTokenizer.load(meta_path)

checkpoint = torch.load(
    ckpt_path,
    map_location=device,
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
            f"checkpoint missing key: {key}. "
            f"Please rerun scripts/train.py to save a new GPT checkpoint."
        )

assert tokenizer.vocab_size == checkpoint["vocab_size"]

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

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
    context,
    max_new_tokens=200,
)

generated_text = tokenizer.decode(
    generated[0].detach().cpu().tolist()
)

print(generated_text)