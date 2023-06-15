from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 1000000
discount = 0.99
tau = 0.005

hidden_dims = [1024, 1024]
critic_lr = 1e-4
actor_lr = 1e-4

num_epoch = 2000
step_per_epoch = 1000
batch_size = 1024

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
random_policy_epoch = 25
max_trajectory_length = 1000

env_type = "dmc"
name = "dmc"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

actor_update_interval = 2
policy_noise = 0.2
noise_clip = 0.5
exploration_noise = 0.1