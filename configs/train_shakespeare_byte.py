dataset = "shakespeare_byte"

batch_size = 32
block_size = 128

max_iters = 500
eval_interval = 100
eval_iters = 20

learning_rate = 1e-3

n_embd = 128
num_heads = 4
n_layer = 4
dropout = 0.2

seed = 42

device = "cuda"

out_dir = "outputs"
ckpt_name = "gpt_shakespeare_byte.pt"

results_dir = "results"
loss_name = "loss_shakespeare_byte.csv"

generate_after_train = True
max_new_tokens = 300