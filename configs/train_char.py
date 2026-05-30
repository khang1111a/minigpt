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

out_dir = "outputs"
ckpt_name = "gpt.pt"

results_dir = "results"
loss_name = "loss.csv"

generate_after_train = True
max_new_tokens = 100