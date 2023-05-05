from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 1000000
discount = 0.99
tau = 0.005
reward_scale = 1.0

alpha = 0.2
auto_alpha = True  # this will dominate alpha

clip_max = 10
target_update_freq = 2
learning_rate = 1e-4
warmup_epoch = 5
random_policy_epoch = 5

actor_hidden_dims = [1024, 1024]
critic_q_hidden_dims = [1024, 1024]
critic_v_hidden_dims = [1024, 1024]

batch_size = 1024
eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
max_trajectory_length = 1000

env_type = "dmc"
name = "dmc"
class wandb(NameSpace):
    entity = None
    project = None
    
debug = False

loss_temperature = 10.0