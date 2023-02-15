from reproduce.iql.config.offline.corl.base import *

task = None

actor_lr = 3e-4
critic_q_lr = 3e-4
critic_v_lr = 3e-4
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005

dropout = None
expectile = 0.9
temperature = 10.0

normalize_obs = True
normalize_reward = True