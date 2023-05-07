from UtilsRL.misc import NameSpace

seed = 42

max_epoch = 1000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
batch_size = 256
max_action = 1.0

normalize_obs = False
normalize_reward = False
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005
learning_rate = 3e-4

discard_last = True

name = "mixed"
class wandb(NameSpace):
    entity = None
    project = None

debug = False


# mixed
temperature = 0.33

agent = "halfcheetah"
quality1 = "medium"
quality2 = "random"
num_data = 1000_000
ratio = 0.5
keep_traj = True
reward_norm = "std"

logstd_hard_clip = False