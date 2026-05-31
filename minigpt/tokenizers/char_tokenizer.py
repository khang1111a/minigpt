import pickle
from pathlib import Path

from minigpt.tokenizers.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    name = "char"

    def __init__(self, text = None, stoi = None, itos = None):
        if text is not None:
            chars = sorted(list(set(text)))

            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}

        elif stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos

        else:
            raise ValueError("You must provide either text or stoi and itos.")
    
    @property
    def vocab_size(self):
        return len(self.stoi)
    
    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])
    
    def save(self, path):
        path = Path(path)

        meta = {
            "tokenizer_type": self.name,
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
        }

        with open(path, "wb") as f:
            pickle.dump(meta, f)
    
    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path, "rb") as f:
            meta = pickle.load(f)

        return cls(
            stoi=meta["stoi"],
            itos=meta["itos"],
        )