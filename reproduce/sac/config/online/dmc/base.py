from reproduce.sac.config.online.base import *

task_type = "dmc"

critic_hidden_dims = [1024, 1024]
critic_lr = 0.0003
actor_hidden_dims = [1024, 1024]
actor_lr = 0.0003
actor_type = "SquashedGaussianActor"

num_epoch = 2000
step_per_epoch = 1000
batch_size = 1024