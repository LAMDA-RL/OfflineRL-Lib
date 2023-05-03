from reproduce.iql.config.d4rl.base import *

task = None

actor_lr = 3e-4
critic_q_lr = 3e-4
critic_v_lr = 3e-4
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005

dropout = 0.1
expextile = 0.7
temperature = 0.5

normalize_obs = False
normalize_reward = True