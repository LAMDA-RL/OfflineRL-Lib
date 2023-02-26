from UtilsRL.misc import NameSpace

seed = 42

betas = [0.9, 0.999]
clip_grad = 0.25
episode_len = 1000
weight_decay = 1e-4

normalize_obs = True
normalize_reward = False

embed_dim = 128
embed_dropout = 0.1
attention_dropout = 0.1
residual_dropout = 0.1
seq_len = 20
num_heads = 1
num_layers = 3
num_workers = 4

max_epoch = 1000
step_per_epoch = 100
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
warmup_steps = 10000

name = "original"
class wandb(NameSpace):
    entity = None
    project = None

debug = False
