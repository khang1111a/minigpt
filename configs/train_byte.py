dataset = "tiny_text_byte"

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

seed = 42

device = "cpu"

out_dir = "outputs"
ckpt_name = "gpt_byte.pt"

results_dir = "results"
loss_name = "loss_byte.csv"

generate_after_train = True
max_new_tokens = 100