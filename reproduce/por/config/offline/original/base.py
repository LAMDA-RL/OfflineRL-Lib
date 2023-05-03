from UtilsRL.misc import NameSpace

seed = 0

pretrain_epoch = 1000
max_epoch = 1000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
batch_size = 256
max_action = 1.0

normalize_obs = False
normalize_reward = True
norm_layer = False

discount = 0.99
tau = 0.005

name = "original"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

actor_hidden_dims = [1024, 1024]
goal_hidden_dims = [256, 256]
critic_v_hidden_dims = [256, 256]
v_expectile = 0.9
actor_lr = 1e-4
critic_v_lr = 1e-4
alpha = 10.0
actor_lr_scheduler_max_steps = 1000_000


variant = "residual"
