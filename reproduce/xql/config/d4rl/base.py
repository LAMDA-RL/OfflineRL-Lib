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

name = "d4rl"
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
