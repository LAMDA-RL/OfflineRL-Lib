from UtilsRL.misc import NameSpace

seed = 42

max_epoch = 1000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
batch_size = 256



discount = 0.9
tau = 0.01

hidden_dims = [64, 64]

dropout = None

normalize_obs = False
normalize_reward = False
actor_opt_decay_schedule = "cosine"


name = "original"
class wandb(NameSpace):
    entity = None
    project = None
    
debug = False
    