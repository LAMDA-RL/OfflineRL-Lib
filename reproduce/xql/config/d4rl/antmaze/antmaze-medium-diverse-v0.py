from reproduce.xql.config.d4rl.antmaze.base import *

task = "antmaze-medium-diverse-v0"

batch_size = 256
double = True
noise_std = 0
max_clip = 7
loss_temperature = 0.6
scale_random_sample = 0

eval_episode = 100
eval_interval = 20

num_v_update = 4