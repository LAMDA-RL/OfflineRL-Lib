from reproduce.xql.config.d4rl.mujoco.base import *

task = "halfcheetah-medium-replay-v2"

batch_size = 256
double = True
noise_std = 0
max_clip = 5
loss_temperature = 1.0
scale_random_sample = 0