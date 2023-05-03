from reproduce.xql.config.d4rl.mujoco.base import *

task = "walker2d-medium-replay-v2"

batch_size = 1024
double = True
noise_std = 0
max_clip = 5
loss_temperature = 5.0
scale_random_sample = 0