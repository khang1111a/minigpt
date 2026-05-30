import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from minigpt.tokenizer import CharTokenizer

data_dir = Path(__file__).parent
input_path = data_dir / "input.txt"

with open(input_path,"r",encoding = "utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

ids = tokenizer.encode(text)
ids = np.array(ids,dtype = np.uint16)

n = int(0.9 * len(ids))

train_ids = ids[:n]
val_ids = ids[n:]

train_text = tokenizer.decode(train_ids)
val_text = tokenizer.decode(val_ids)

train_path = data_dir / "train.bin"
val_path = data_dir / "val.bin"

train_ids.tofile(train_path)
val_ids.tofile(val_path)

meta_path = data_dir / "meta.pkl"

tokenizer.save(meta_path)

loaded_train = np.fromfile(train_path, dtype=np.uint16)
loaded_val = np.fromfile(val_path, dtype=np.uint16)

loaded_tokenizer = CharTokenizer.load(meta_path)
