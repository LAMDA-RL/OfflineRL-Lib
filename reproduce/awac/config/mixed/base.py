from UtilsRL.misc import NameSpace

seed = 42
device = None
task = None

max_epoch = 1000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
max_action = 1.0

hidden_dims = [256, 256, 256]
discount = 0.99
tau = 0.005
actor_weight_decay = 0.0

actor_lr = 3e-4
critic_lr = 3e-4

batch_size = 256

normalize_obs = True
normalize_reward = False

name = "mixed"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

# define the aw lambda
aw_lambda = 1/3

# mixed
agent = "halfcheetah"
quality1 = "medium"
quality2 = "random"
num_data = 1000_000
ratio = 0.5
keep_traj = True
reward_norm = "std"