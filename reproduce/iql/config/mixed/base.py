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

actor_opt_decay_schedule = "cosine"
conditioned_logstd = False
policy_logstd_min = -5.0

name = "mixed"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

actor_lr = 3e-4
critic_q_lr = 3e-4
critic_v_lr = 3e-4
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005

dropout = None
expectile = 0.7
temperature = 3.0

normalize_obs = False
normalize_reward = True

# mixed

# mixed
agent = "halfcheetah"
quality1 = "medium"
quality2 = "random"
num_data = 1000_000
ratio = 0.5
keep_traj = True
reward_norm = "std"

iql_deterministic = False