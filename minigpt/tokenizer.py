import pickle

class CharTokenizer:
    def __init__(self,text):
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self,text):
        return [self.stoi[ch] for ch in text]
    
    def decode(self,ids):
        return "".join([self.itos[i] for i in ids])
    
    def save(self,path):
        meta = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
        }

        with open(path,"wb") as f:
            pickle.dump(meta,f)

    @classmethod
    def load(cls,path):
        with open(path,"rb") as f:
            meta = pickle.load(f)

        tokenizer = cls.__new__(cls)

        tokenizer.stoi = meta["stoi"]
        tokenizer.itos = meta["itos"]
        tokenizer.vocab_size = meta["vocab_size"]

        return tokenizer

if __name__ == "__main__":
    text = "hello world"

    tokenizer = CharTokenizer(text)

    print("stoi:",tokenizer.stoi)
    print("itos:",tokenizer.itos)
    print("vocab size:",tokenizer.vocab_size)

    ids = tokenizer.encode("hello world")
    print("encode:",ids)

    decoded = tokenizer.decode(ids)
    print("decoded:", decoded)

    tokenizer.save("meta.pkl")

    tokenizer2 = CharTokenizer.load("meta.pkl")

    ids2 = tokenizer2.encode("hello")
    print("loaded encode:", ids2)

    decoded2 = tokenizer2.decode(ids2)
    print("loaded decoded:", decoded2)

    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
    assert tokenizer2.decode(tokenizer2.encode("hello")) == "hello"
    assert tokenizer.encode("hello") == tokenizer2.encode("hello")

    print("All tests passed.")