from reproduce.xql.config.offline.antmaze.base import *

task = "antmaze-large-diverse-v0"

batch_size = 1024
double = True
noise_std = 0
max_clip = 5
loss_temperature = 0.55
scale_random_sample = 0

eval_episode = 100
eval_interval = 20