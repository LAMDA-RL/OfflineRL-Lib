from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 1000000
discount = 0.99
tau = 0.005
alpha = 0.2
auto_alpha = True
reward_scale = 1.0

critic_hidden_dims = [1024, 1024]
critic_lr = 1e-4
actor_hidden_dims = [1024, 1024]
actor_lr = 1e-4

alpha_lr = 1e-4

num_epoch = 2000
step_per_epoch = 1000
batch_size = 1024

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
warmup_epoch = 5
random_policy_epoch = 5
max_trajectory_length = 1000

policy_logstd_min = -5
policy_logstd_max = 2

target_update_freq = 2

env_type = "dmc"
name = "dmc"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

critic_q_num = 2