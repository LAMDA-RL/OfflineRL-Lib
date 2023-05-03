from reproduce.xql.config.d4rl.mujoco.base import *

task = "halfcheetah-medium-v2"

batch_size = 1024
double = True
noise_std = 0.1
max_clip = 7
loss_temperature = 1.0
scale_random_sample = 0