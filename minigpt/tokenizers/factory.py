import pickle
from pathlib import Path

from minigpt.tokenizers.char_tokenizer import CharTokenizer
from minigpt.tokenizers.byte_tokenizer import ByteTokenizer


def build_tokenizer(name, text = None):
    name = name.lower()

    if name == "char":
        if text is None:
            raise ValueError("CharTokenizer requires text.")
        return CharTokenizer(text = text)

    if name == "byte":
        return ByteTokenizer()

    raise ValueError(f"Unknown tokenizer: {name}")

def load_tokenizer(path):
    path = Path(path)

    with open(path, "rb") as f:
        meta = pickle.load(f)

    tokenizer_type = meta.get("tokenizer_type")

    if tokenizer_type == "char":
        return CharTokenizer.load(path)

    if tokenizer_type == "byte":
        return ByteTokenizer.load(path)

    raise ValueError(f"Unknown tokenizer type in meta file: {tokenizer_type}")