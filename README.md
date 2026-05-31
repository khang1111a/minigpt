# MiniGPT

MiniGPT 是一个基于 PyTorch 从零实现的轻量级 GPT 语言模型训练框架。项目目标是以尽可能清晰、模块化的方式复现 GPT-style decoder-only Transformer 的核心流程，包括 tokenizer、数据预处理、批量采样、模型结构、训练评估、checkpoint 管理与自回归文本生成。

本项目定位为学习型与实验型框架，重点不在于追求大规模语言模型性能，而在于完整理解并实现 GPT 语言模型的主要工程链路。

## 功能特性

* 从零实现 GPT-style Transformer decoder。
* 支持字符级 tokenizer 与 byte-level tokenizer。
* 支持统一数据预处理入口，将原始文本转换为二进制 token 数据。
* 支持基于配置文件的训练流程。
* 实现 token embedding、position embedding、causal self-attention、multi-head attention、feed-forward network、residual connection、LayerNorm 与 dropout。
* 支持 train / validation loss 估计与 CSV 日志记录。
* 支持 checkpoint 保存与加载。
* 支持 prompt 条件生成、temperature 采样、top-k 采样与随机种子控制。
* 支持 loss 曲线可视化。
* 保留旧 tokenizer 入口以兼容早期代码。

## 项目结构

```text
minigpt/
├── configs/
│   ├── train_char.py
│   └── train_byte.py
├── data/
│   ├── tiny_text/
│   │   ├── input.txt
│   │   └── prepare.py
│   ├── tiny_text_char/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   └── tiny_text_byte/
│       ├── train.bin
│       ├── val.bin
│       └── meta.pkl
├── minigpt/
│   ├── dataset.py
│   ├── model.py
│   ├── tokenizer.py
│   └── tokenizers/
│       ├── __init__.py
│       ├── base.py
│       ├── char_tokenizer.py
│       ├── byte_tokenizer.py
│       └── factory.py
├── outputs/
│   ├── gpt_char.pt
│   └── gpt_byte.pt
├── results/
│   ├── loss_char.csv
│   ├── loss_char.png
│   ├── loss_byte.csv
│   └── loss_byte.png
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── sample.py
│   └── plot_loss.py
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

依赖项：

```text
torch
numpy
matplotlib
```

## 快速开始

### 1. 准备字符级数据集

```powershell
python scripts\prepare_data.py --input data/tiny_text/input.txt --out data/tiny_text_char --tokenizer char
```

### 2. 训练字符级 GPT 模型

```powershell
python scripts\train.py --config configs/train_char.py
```

### 3. 使用 prompt 生成文本

```powershell
python scripts\sample.py --config configs/train_char.py --prompt "this is" --temperature 0.8 --top-k 10 --max-new-tokens 200
```

### 4. 绘制训练损失曲线

```powershell
python scripts\plot_loss.py --loss results/loss_char.csv
```

## Tokenizer 系统

项目支持模块化 tokenizer。所有 tokenizer 遵循统一接口：

```python
encode(text)
decode(ids)
vocab_size
save(path)
load(path)
```

相关代码位于：

```text
minigpt/tokenizers/
```

推荐使用统一入口：

```python
from minigpt.tokenizers import build_tokenizer, load_tokenizer
```

旧入口 `minigpt/tokenizer.py` 保留为兼容层，早期代码仍可使用：

```python
from minigpt.tokenizer import CharTokenizer
```

### CharTokenizer

`CharTokenizer` 根据训练文本中出现过的字符构建词表。

优点：

* 实现简单，便于理解。
* 适合调试和教学。
* 小文本上训练速度快。

局限：

* 只能编码训练语料中出现过的字符。
* 更换语料后通常需要重新构建词表。

### ByteTokenizer

`ByteTokenizer` 将文本编码为 UTF-8 byte 序列，词表大小固定为 256。

优点：

* 不需要训练词表。
* 支持任意 UTF-8 文本。
* 不存在未知字符问题。
* 适合多语言文本实验。

局限：

* 中文等非 ASCII 字符会被拆分为多个 byte。
* 模型训练不足时可能生成非法 UTF-8 byte，解码后显示为 `�`。

## 数据预处理

统一数据预处理入口为：

```text
scripts/prepare_data.py
```

它将原始文本转换为：

```text
train.bin
val.bin
meta.pkl
```

### 字符级数据准备

```powershell
python scripts\prepare_data.py --input data/tiny_text/input.txt --out data/tiny_text_char --tokenizer char
```

### Byte-level 数据准备

```powershell
python scripts\prepare_data.py --input data/tiny_text/input.txt --out data/tiny_text_byte --tokenizer byte
```

输出文件说明：

```text
train.bin    训练 token 序列
val.bin      验证 token 序列
meta.pkl     tokenizer 元数据
```

早期脚本 `data/tiny_text/prepare.py` 保留用于兼容。新的实验建议统一使用 `scripts/prepare_data.py`。

## 模型结构

核心模型位于：

```text
minigpt/model.py
```

当前模型采用 GPT-style decoder-only Transformer 架构：

```text
token embedding
+ position embedding
↓
Transformer Block × n_layer
↓
final LayerNorm
↓
linear language modeling head
↓
next-token logits
```

每个 Transformer Block 包含：

```text
LayerNorm
Multi-Head Causal Self-Attention
Residual Connection
LayerNorm
FeedForward Network
Residual Connection
```

主要模块包括：

* `Head`：单个 causal self-attention head。
* `MultiHeadAttention`：多头注意力模块，包含多个 attention head 与输出投影。
* `FeedForward`：两层 MLP，中间维度为 `4 * n_embd`。
* `Block`：标准 Transformer decoder block。
* `GPTLanguageModel`：完整自回归语言模型，支持 loss 计算与文本生成。

## 训练目标

MiniGPT 使用 next-token prediction 作为训练目标。

给定输入序列：

```text
x = [token_0, token_1, ..., token_t]
```

目标序列为右移一位后的 token 序列：

```text
y = [token_1, token_2, ..., token_{t+1}]
```

模型在每个位置预测下一个 token。由于使用 causal mask，每个位置只能关注自身及之前的 token，不能看到未来 token。

## 模型训练

训练脚本为：

```text
scripts/train.py
```

字符级训练：

```powershell
python scripts\train.py --config configs/train_char.py
```

Byte-level 训练：

```powershell
python scripts\train.py --config configs/train_byte.py
```

配置文件示例：

```python
dataset = "tiny_text_char"

batch_size = 4
block_size = 8

max_iters = 200
eval_interval = 20
eval_iters = 10

learning_rate = 1e-3

n_embd = 32
num_heads = 4
n_layer = 2
dropout = 0.2

seed = 1337
device = "cpu"

out_dir = "outputs"
ckpt_name = "gpt_char.pt"

results_dir = "results"
loss_name = "loss_char.csv"

generate_after_train = True
max_new_tokens = 100
```

训练完成后会保存：

```text
outputs/gpt_char.pt
results/loss_char.csv
```

Byte-level 训练对应保存：

```text
outputs/gpt_byte.pt
results/loss_byte.csv
```

## 文本生成

生成脚本为：

```text
scripts/sample.py
```

字符级生成：

```powershell
python scripts\sample.py --config configs/train_char.py --prompt "this is" --max-new-tokens 200
```

带采样控制的生成：

```powershell
python scripts\sample.py --config configs/train_char.py --prompt "this is" --temperature 0.8 --top-k 10 --seed 1337
```

Byte-level 生成：

```powershell
python scripts\sample.py --config configs/train_byte.py --prompt "this is" --temperature 0.7 --top-k 50
```

参数说明：

* `--config`：训练配置文件路径。
* `--prompt`：生成起始文本。
* `--max-new-tokens`：最大生成 token 数。
* `--temperature`：采样温度，数值越低生成越保守，数值越高生成越随机。
* `--top-k`：仅从概率最高的 k 个 token 中采样。
* `--seed`：采样随机种子，用于复现实验结果。

## Loss 曲线绘制

绘制字符级模型 loss 曲线：

```powershell
python scripts\plot_loss.py --loss results/loss_char.csv
```

绘制 byte-level 模型 loss 曲线：

```powershell
python scripts\plot_loss.py --loss results/loss_byte.csv
```

自定义输出路径：

```powershell
python scripts\plot_loss.py --loss results/loss_byte.csv --out results/byte_curve.png
```

`plot_loss.py` 同时兼容当前格式：

```csv
step,train_loss,val_loss
```

以及早期格式：

```csv
step,loss
```


## 常见问题

### checkpoint 文件不存在

如果运行 `sample.py` 时提示 checkpoint 文件不存在，说明尚未使用对应配置训练模型。

运行：

```powershell
python scripts\train.py --config configs/train_char.py
```

或：

```powershell
python scripts\train.py --config configs/train_byte.py
```

### checkpoint 缺少字段

通常说明 checkpoint 来自旧版本训练脚本。重新运行当前版本训练脚本即可。

```powershell
python scripts\train.py --config configs/train_char.py
```

### CharTokenizer 遇到未知字符

`CharTokenizer` 只能编码训练语料中出现过的字符。如果 prompt 中含有未见字符，会报错。

解决方式：

```text
1. 使用训练语料中出现过的字符作为 prompt。
2. 使用更大的文本重新 prepare 数据。
3. 改用 ByteTokenizer。
```

### ByteTokenizer 输出 `�`

Byte-level 模型在训练不足时可能生成非法 UTF-8 byte 序列，解码后会显示为 `�`。

可以尝试降低温度并使用 top-k 采样：

```powershell
python scripts\sample.py --config configs/train_byte.py --temperature 0.7 --top-k 50
```

也可以通过增加训练数据与训练步数改善生成质量。

### val loss 为空

如果验证集 token 数量小于或等于 `block_size`，则无法构造验证 batch，训练脚本会跳过 val loss。

对于 `tiny_text` 这种极小数据集，这是正常现象。使用更大的语料后即可正常记录 val loss。
