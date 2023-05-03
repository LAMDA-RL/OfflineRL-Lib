from UtilsRL.misc import NameSpace

seed = 42

max_epoch = 1000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
batch_size = 1024
num_v_update = 1

use_log_loss = False
noise_std = 0

name = "mixed"
class wandb(NameSpace):
    entity = None
    project = None

conditioned_logstd = False
policy_logstd_min = -5.0
max_action = 1.0
normalize_obs = False
normalize_reward = True
actor_opt_decay_schedule = "cosine"

debug = False

# ===== mujoco ======
actor_lr = 3e-4
critic_q_lr = 3e-4
critic_v_lr = 3e-4
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005

aw_temperature = 3.0
norm_layer = True
dropout = None
value_dropout = None

actor_opt_decay_schedule = "cosine"

# ====== mujoco consistent ======
batch_size = 256
double = True
noise_std = 0
max_clip = 7
loss_temperature = 2.0
scale_random_sample = 0

# mixed
agent = "halfcheetah"
quality1 = "medium"
quality2 = "random"
num_data = 1000_000
ratio = 0.5
keep_traj = True
reward_norm = "std"
