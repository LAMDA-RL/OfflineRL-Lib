from reproduce.xql.config.d4rl.antmaze.base import *

task = "antmaze-umaze-diverse-v0"

batch_size = 256
double = True
noise_std = 0
max_clip = 7
loss_temperature = 5.0
scale_random_sample = 1

eval_episode = 100
eval_interval = 20