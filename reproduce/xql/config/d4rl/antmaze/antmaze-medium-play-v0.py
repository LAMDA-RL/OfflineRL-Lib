from reproduce.xql.config.d4rl.antmaze.base import *

task = "antmaze-medium-play-v0"

batch_size = 256
double = True
noise_std = 0
max_clip = 5
loss_temperature = 0.8
scale_random_sample = 0

eval_episode = 100
eval_interval = 20