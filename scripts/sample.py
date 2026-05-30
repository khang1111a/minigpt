import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.model import BigramLanguageModel
from minigpt.tokenizer import CharTokenizer


data_dir = ROOT / "data" / "tiny_text"
output_dir = ROOT / "outputs"

meta_path = data_dir / "meta.pkl"
ckpt_path = output_dir / "bigram.pt"

tokenizer = CharTokenizer.load(meta_path)
checkpoint = torch.load(ckpt_path, map_location="cpu")

assert tokenizer.vocab_size == checkpoint["vocab_size"]

model = BigramLanguageModel(checkpoint["vocab_size"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

context = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(context, max_new_tokens=100)

generated_text = tokenizer.decode(generated[0].tolist())

print(generated_text)