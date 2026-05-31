import pickle
from pathlib import Path

from minigpt.tokenizers.base import BaseTokenizer


class ByteTokenizer(BaseTokenizer):
    name = "byte"
    
    def __init__(self):
        super().__init__()

    @property
    def vocab_size(self):
        return 256
    
    def encode(self,text):
        return list(text.encode("utf-8"))
    
    def decode(self, ids):
        ids = [int(i) for i in ids]
        return bytes(ids).decode("utf-8", errors="replace")
    
    def save(self, path):
        path = Path(path)

        meta = {
            "tokenizer_type": self.name,
            "vocab_size": self.vocab_size,

        }

        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path, "rb") as f:
            meta = pickle.load(f)

        if meta["tokenizer_type"] != cls.name:
            raise ValueError(
                f"Tokenizer type mismatch: expected {cls.name}, "
                f"got {meta['tokenizer_type']}"
            )

        return cls()