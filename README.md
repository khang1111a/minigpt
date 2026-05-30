# MiniGPT

这是一个用于学习 GPT 语言模型核心机制的最小实现。当前版本实现了字符级 tokenizer、二进制数据集准备、批量采样、Transformer decoder 语言模型、训练脚本、文本生成脚本，以及训练损失曲线绘制。

## 功能概览

- 字符级分词器：从训练文本中构建字符表，支持 `encode` / `decode` 和 `meta.pkl` 持久化。
- 数据准备：把 `input.txt` 编码为 `train.bin` 和 `val.bin`，并按 90% / 10% 划分训练集和验证集。
- GPT 模型：包含 token embedding、position embedding、多头 causal self-attention、前馈网络、残差连接和 LayerNorm。
- 训练流程：读取配置文件训练模型，保存 loss 记录和 checkpoint。
- 文本生成：从保存的 `outputs/gpt.pt` 加载模型并采样生成文本。
- 可视化：把 `results/loss.csv` 绘制为 `results/loss.png`。

## 项目结构

```text
minigpt/
|-- configs/
|   `-- train_char.py        # 训练超参数和输出路径配置
|-- data/
|   `-- tiny_text/
|       |-- input.txt        # 原始文本
|       |-- prepare.py       # 数据集准备脚本
|       |-- train.bin        # 训练 token 数据
|       |-- val.bin          # 验证 token 数据
|       `-- meta.pkl         # 字符表和 tokenizer 元数据
|-- minigpt/
|   |-- dataset.py           # 批量采样
|   |-- model.py             # GPT 语言模型
|   `-- tokenizer.py         # 字符级 tokenizer
|-- outputs/
|   |-- bigram.pt
|   `-- gpt.pt               # 当前 GPT checkpoint
|-- results/
|   |-- loss.csv             # 训练 loss 记录
|   `-- loss.png             # loss 曲线图
|-- scripts/
|   |-- plot_loss.py         # 绘制 loss 曲线
|   |-- sample.py            # 加载 checkpoint 生成文本
|   `-- train.py             # 训练入口
`-- requirements.txt
```

## 环境准备

建议使用 Python 虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

依赖项：

- `torch`
- `numpy`
- `matplotlib`

## 数据准备

当前数据集位于 `data/tiny_text/`。如果修改了 `data/tiny_text/input.txt`，需要重新生成二进制数据和 tokenizer 元数据：

```powershell
python data\tiny_text\prepare.py
```

该脚本会生成：

- `data/tiny_text/train.bin`
- `data/tiny_text/val.bin`
- `data/tiny_text/meta.pkl`

## 训练模型

训练入口：

```powershell
python scripts\train.py
```

训练脚本会读取 `configs/train_char.py` 中的配置，默认参数如下：

```python
dataset = "tiny_text"
batch_size = 4
block_size = 8
max_iters = 200
eval_interval = 20
learning_rate = 1e-3
n_embd = 32
num_heads = 4
n_layer = 2
dropout = 0.2
device = "cpu"
```

训练结束后会输出：

- `results/loss.csv`：按 `eval_interval` 记录的训练 loss。
- `outputs/gpt.pt`：模型 checkpoint，包含模型权重和结构参数。

如果 `generate_after_train = True`，训练结束后会直接生成一段文本。

## 生成文本

使用当前保存的 GPT checkpoint：

```powershell
python scripts\sample.py
```

`sample.py` 会加载：

- `data/tiny_text/meta.pkl`
- `outputs/gpt.pt`

然后从初始 token 开始采样生成 200 个新 token。

## 绘制损失曲线

训练后可以把 `results/loss.csv` 绘制成图片：

```powershell
python scripts\plot_loss.py
```

输出文件：

```text
results/loss.png
```

## 模型实现说明

核心模型在 `minigpt/model.py` 中：

- `Head`：单个 causal self-attention 头，使用下三角 mask 防止当前位置看到未来 token。
- `MultiHeadAttention`：并行多个 attention head，再通过线性层投影回 embedding 维度。
- `FeedForward`：两层 MLP，中间维度为 `4 * n_embd`。
- `Block`：一个 Transformer block，包含 self-attention、feed-forward、残差连接和 LayerNorm。
- `GPTLanguageModel`：完整字符级语言模型，支持前向计算 loss 和自回归生成。

训练目标是 next-token prediction：输入序列 `x` 的每个位置预测下一个字符 `y`。

## 调整配置

常用配置在 `configs/train_char.py`：

- 想扩大上下文长度：增大 `block_size`。
- 想增加模型容量：增大 `n_embd`、`num_heads` 或 `n_layer`。
- 想训练更久：增大 `max_iters`。
- 想使用 GPU：如果已安装支持 CUDA 的 PyTorch，可把 `device` 改为 `"cuda"`。

注意：`sample.py` 当前固定读取 `data/tiny_text` 和 `outputs/gpt.pt`。如果更换数据集或 checkpoint 路径，需要同步修改该脚本。

## 常见问题

### checkpoint 缺少字段

如果运行 `scripts/sample.py` 时提示 checkpoint 缺少字段，说明当前 checkpoint 不是新训练脚本保存的格式。重新运行：

```powershell
python scripts\train.py
```

### loss 文件不存在

如果运行 `scripts/plot_loss.py` 时提示找不到 `results/loss.csv`，需要先训练一次模型：

```powershell
python scripts\train.py
```

### 字符编码报错

如果更换了 `input.txt`，请确保文本使用 UTF-8 编码保存，然后重新运行数据准备脚本。
