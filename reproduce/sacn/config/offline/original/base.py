from UtilsRL.misc import NameSpace

seed = 42
device = None
task = None

max_epoch = 3000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
max_action = 1.0

hidden_dims = [256, 256, 256]
discount = 0.99
tau = 0.005
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
auto_alpha = True
batch_size = 256
policy_logstd_min = -5

normalize_obs = False
normalize_reward = False

do_reverse_update = True

name = "original"
class wandb(NameSpace):
    entity = None
    project = None

debug = False
