from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 1000000
gamma = 0.99
tau = 0.005
alpha = 0.2
reward_scale = 1.0

critic_hidden_dims = [256, 256]
critic_lr = 0.0003
actor_hidden_dims = [256, 256]
actor_lr = 0.0003
actor_type = "SquashedGaussianActor"

auto_alpha = True
alpha_lr = 0.0003

num_epoch = 3000
step_per_epoch = 1000
batch_size = 256

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
save_interval = 50
warmup_epoch = 2
random_policy_epoch = 5
max_trajectory_length = 1000

name = "original"
class wandb(NameSpace):
    entity = None
    project = None

debug = False
