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
* 支持 checkpoint 保存与加载，包含 optimizer 状态、训练 step 与 best validation loss。
* 支持 prompt 条件生成、temperature 采样、top-k 采样与随机种子控制。
* 支持交互式文本生成控制台。
* 支持 loss 曲线可视化。
* 支持 pytest 基础测试，覆盖 tokenizer 与数据预处理链路。
* 保留旧 tokenizer 入口以兼容早期代码。

## 项目结构

```text
minigpt/
├── configs/
│   ├── train_char.py
│   ├── train_byte.py
│   ├── train_shakespeare_char.py
│   └── train_shakespeare_byte.py
├── data/
│   ├── tiny_text/
│   │   ├── input.txt
│   │   └── prepare.py
│   ├── tiny_text_char/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   ├── tiny_text_byte/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   ├── shakespeare/
│   │   └── input.txt
│   ├── shakespeare_char/
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   └── shakespeare_byte/
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
│   ├── loss_byte.png
│   ├── loss_shakespeare_char.csv
│   ├── loss_shakespeare_char.png
│   ├── loss_shakespeare_byte.csv
│   └── loss_shakespeare_byte.png
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── sample.py
│   ├── console.py
│   └── plot_loss.py
├── tests/
│   ├── test_cli.py
│   ├── test_prepare_data.py
│   └── test_tokenizers.py
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
datasets
tqdm
pytest
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

### 5. 运行测试

```powershell
python -m pytest --basetemp .pytest_tmp
```

`--basetemp .pytest_tmp` 用于在项目目录下创建 pytest 临时目录，避免 Windows 系统临时目录权限问题。

### 6. 运行 Shakespeare 实验

项目已经包含 tiny Shakespeare 的 char 与 byte 两组实验配置：

```powershell
python scripts\train.py --config configs/train_shakespeare_char.py
python scripts\train.py --config configs/train_shakespeare_byte.py
```

训练完成后可以生成文本：

```powershell
python scripts\sample.py --config configs/train_shakespeare_char.py --prompt "ROMEO:" --temperature 0.8 --top-k 20 --max-new-tokens 500 --seed 42
python scripts\sample.py --config configs/train_shakespeare_byte.py --prompt "ROMEO:" --temperature 0.8 --top-k 40 --max-new-tokens 500 --seed 42
```

也可以显式加载 best checkpoint：

```powershell
python scripts\sample.py --config configs/train_shakespeare_char.py --ckpt outputs/gpt_shakespeare_char_best.pt --prompt "ROMEO:" --temperature 0.8 --top-k 20 --max-new-tokens 500 --seed 42
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

### Shakespeare 数据准备

字符级数据：

```powershell
python scripts\prepare_data.py --input data/shakespeare/input.txt --out data/shakespeare_char --tokenizer char --train-ratio 0.9
```

Byte-level 数据：

```powershell
python scripts\prepare_data.py --input data/shakespeare/input.txt --out data/shakespeare_byte --tokenizer byte --train-ratio 0.9
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

Shakespeare 字符级训练：

```powershell
python scripts\train.py --config configs/train_shakespeare_char.py
```

Shakespeare byte-level 训练：

```powershell
python scripts\train.py --config configs/train_shakespeare_byte.py
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
outputs/gpt_char_best.pt
results/loss_char.csv
```

Byte-level 训练对应保存：

```text
outputs/gpt_byte.pt
outputs/gpt_byte_best.pt
results/loss_byte.csv
```

checkpoint 文件属于训练产物，默认不建议提交到 Git。当前 `.gitignore` 会忽略 `outputs/*.pt`。

当前 checkpoint 会保存：

```text
model_state_dict
optimizer_state_dict
step
best_val_loss
vocab_size
block_size
n_embd
num_heads
n_layer
dropout
seed
dataset
learning_rate
batch_size
```

其中 `outputs/<name>.pt` 是 latest/final checkpoint，`outputs/<name>_best.pt` 是验证集 loss 最低时保存的 best checkpoint。

### 断点续训

训练脚本支持从新格式 checkpoint 恢复训练：

```powershell
python scripts\train.py --config configs/train_shakespeare_char.py --resume outputs/gpt_shakespeare_char.pt
```

`--resume` 会恢复模型参数、AdamW optimizer 状态、当前 step 和 `best_val_loss`。配置文件里的 `max_iters` 表示总目标步数，而不是额外训练步数。例如 checkpoint 保存于 `step = 499`，配置中 `max_iters = 800`，恢复后会继续训练 `500` 到 `799`。

旧格式 checkpoint 如果缺少 `optimizer_state_dict` 或 `step`，不能用于严格断点续训，需要重新训练生成新格式 checkpoint。

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

Shakespeare 生成：

```powershell
python scripts\sample.py --config configs/train_shakespeare_char.py --prompt "ROMEO:" --temperature 0.8 --top-k 20 --max-new-tokens 500 --seed 42
python scripts\sample.py --config configs/train_shakespeare_byte.py --prompt "ROMEO:" --temperature 0.8 --top-k 40 --max-new-tokens 500 --seed 42
```

指定 checkpoint 生成：

```powershell
python scripts\sample.py --config configs/train_shakespeare_char.py --ckpt outputs/gpt_shakespeare_char_best.pt --prompt "ROMEO:" --temperature 0.8 --top-k 20 --max-new-tokens 500 --seed 42
```

参数说明：

* `--config`：训练配置文件路径。
* `--ckpt`：可选 checkpoint 路径；不传时读取配置文件里的 `out_dir` 和 `ckpt_name`。
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

绘制 Shakespeare 实验曲线：

```powershell
python scripts\plot_loss.py --loss results/loss_shakespeare_char.csv --out results/loss_shakespeare_char.png
python scripts\plot_loss.py --loss results/loss_shakespeare_byte.csv --out results/loss_shakespeare_byte.png
```

`plot_loss.py` 同时兼容当前格式：

```csv
step,train_loss,val_loss
```

以及早期格式：

```csv
step,loss
```

## 当前实验结果

当前仓库已经完成 tiny_text debug 闭环，以及 tiny Shakespeare 上的 char / byte tokenizer 对比实验。

| 数据集 | tokenizer | vocab size | train tokens | val tokens | 配置文件 | step 400 val loss |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| tiny_text | char | 17 | 34 | 4 | `configs/train_char.py` | N/A |
| tiny_text | byte | 256 | 34 | 4 | `configs/train_byte.py` | N/A |
| shakespeare | char | 65 | 1,003,854 | 111,540 | `configs/train_shakespeare_char.py` | 2.2018 |
| shakespeare | byte | 256 | 1,003,854 | 111,540 | `configs/train_shakespeare_byte.py` | 2.2388 |

`tiny_text` 的验证集太短，主要用于快速检查 tokenizer、dataset、model、train、sample 是否能跑通。Shakespeare 实验使用相同模型规模训练 500 step，每 100 step 记录一次 train / validation loss；当前结果说明 char 与 byte 两条 tokenizer 管线都能完整训练、生成和绘图。

500 step 仍然是短训，生成文本通常会出现英文形状、换行和角色台词格式，但还会包含大量伪词。若要提升生成质量，可以增加 `max_iters`，或后续接入更大的模型与更长训练。

## Roadmap

项目当前已完成基础训练框架、模块化 tokenizer、Shakespeare 基准实验和训练恢复能力。后续工作将围绕实验可复现性、评估体系、交互体验和参数高效微调能力逐步推进。

| 阶段 | 状态 | 目标 | 交付内容 |
| --- | --- | --- | --- |
| v0.2 基础稳定 | 已完成 | 稳定最小训练闭环，降低后续迭代风险 | `.gitignore`、pytest 测试、tokenizer round-trip 测试、数据预处理 smoke test、训练脚本清理 |
| 多数据集支持 | 已完成 | 将项目从 tiny_text debug 数据扩展到真实小语料实验 | tiny Shakespeare 数据接入、char / byte 数据预处理、两组训练配置、loss CSV 与曲线产物 |
| 训练工程化 | 基本完成 | 提升训练过程的可恢复性和实验可追踪性 | best checkpoint、checkpoint metadata、`sample.py --ckpt`、`train.py --resume` |
| 评估与实验对比 | 下一阶段 | 建立统一评估入口，支持不同 tokenizer 与 checkpoint 的量化比较 | `scripts/eval.py`、validation loss、perplexity、char vs byte 对比表、README 实验结果更新 |
| 交互生成增强 | 规划中 | 提升模型演示和人工观察效率 | `scripts/console.py` 支持 checkpoint 选择、生成参数展示、生成日志保存 |
| Simple BPE tokenizer | 规划中 | 支持子词级 tokenizer 实验，补齐 char / byte / BPE 对比链路 | 简化 BPE 训练、merges 持久化、encode/decode、数据预处理集成、基础测试 |
| LoRA / Adaptive LoRA | 后续阶段 | 在稳定训练与评估体系上开展参数高效微调实验 | LoRA 模块、rank 配置、full fine-tune vs LoRA 对比、adaptive rank 实验记录 |

近期优先级：

```text
1. 提交 checkpoint / resume 相关 README 更新。
2. 实现 scripts/eval.py，支持通过 config + checkpoint 计算 validation loss 与 perplexity。
3. 使用 eval.py 重新评估 shakespeare_char 与 shakespeare_byte，并更新实验结果表。
4. 增强 scripts/console.py，使其支持 best checkpoint 和生成日志记录。
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

通常说明 checkpoint 来自旧版本训练脚本。生成文本时可以尝试重新训练；断点续训时必须使用包含 `optimizer_state_dict` 和 `step` 的新格式 checkpoint。

```powershell
python scripts\train.py --config configs/train_char.py
```

如果 resume 时报错缺少字段：

```text
checkpoint missing key required for resume: optimizer_state_dict
```

说明该 checkpoint 只能用于旧式加载或需要重新训练生成新格式 checkpoint，不能严格恢复 AdamW 状态。

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
