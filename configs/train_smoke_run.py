dataset = "tiny_text_char"

batch_size = 4
block_size = 8

max_iters = 2
eval_interval = 1
eval_iters = 1

learning_rate = 1e-3

n_embd = 16
num_heads = 4
n_layer = 1
dropout = 0.1

seed = 42
device = "cpu"

out_dir = "outputs"
ckpt_name = "gpt_smoke.pt"

results_dir = "results"
loss_name = "loss_smoke.csv"

generate_after_train = False
max_new_tokens = 20