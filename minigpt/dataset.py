from pathlib import Path
import numpy as np
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minigpt.tokenizer import CharTokenizer

data_dir = Path(__file__).resolve().parents[1] / "data" / "tiny_text"
train_path = data_dir / "train.bin"
val_path = data_dir / "val.bin"
meta_path = data_dir / "meta.pkl"

train_data = np.fromfile(train_path, dtype=np.uint16)
val_data = np.fromfile(val_path, dtype=np.uint16)


def get_batch(split, train_data, val_data, batch_size, block_size):
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be 'train' or 'val'")
    
    if len(data) <= block_size:
        raise ValueError(
            f"data length must be larger than block_size, "
            f"got len(data)={len(data)}, block_size={block_size}"
        )
    
    ix = torch.randint(0, len(data) - block_size, (batch_size,))

    x_list = []
    y_list = []

    for i in ix:
        i = i.item()

        x_i = data[i:i + block_size]
        y_i = data[i + 1:i + block_size + 1]

        x_list.append(x_i)
        y_list.append(y_i)

    x = torch.stack([
        torch.tensor(x_i,dtype = torch.long)
        for x_i in x_list
    ])

    y = torch.stack([
        torch.tensor(y_i,dtype = torch.long)
        for y_i in y_list
    ])

    return x, y



if __name__ == "__main__":
    batch_size = 4
    block_size = 8
    x, y = get_batch(
        split="train",
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        block_size=block_size,
    )

    tokenizer = CharTokenizer.load(meta_path)

