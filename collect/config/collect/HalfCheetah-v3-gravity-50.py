from collect.config.collect.base import *

task = "HalfCheetah-v3"
overwrite_args = {"gravity": 0.5}
do_scale = True
num_epoch = 1000
name = "gravity-50"
actor_policy_ckpt_list = [100, 200, 300, 400, 500]