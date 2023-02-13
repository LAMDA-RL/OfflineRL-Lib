from reproduce.xql.config.offline.base import *

actor_lr = 3e-4
critic_q_lr = 3e-4
critic_v_lr = 3e-4
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005


aw_temperature = 10.0
norm_layer = True
dropout = None
value_dropout = 0.5

actor_opt_decay_schedule = None