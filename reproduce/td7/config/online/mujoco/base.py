from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 1000000
discount = 0.99
max_action = 1.0

num_epoch = 5000
step_per_epoch = 1000
batch_size = 256

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
max_trajectory_length = 1000

env_type = "mujoco"
name = "mujoco"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

actor_lr = critic_lr = encoder_lr = 3e-4

embedding_dim = 256
hidden_dim = 256

actor_update_interval = 2
target_update_interval = 250
policy_noise = 0.2
noise_clip = 0.5
exploration_noise = 0.1
step_before_training = 25_000

use_checkpoint = False