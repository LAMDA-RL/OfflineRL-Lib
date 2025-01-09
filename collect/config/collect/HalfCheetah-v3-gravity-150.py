from collect.config.collect.base import *

task = "HalfCheetah-v3"
overwrite_args = {"gravity": 1.5}
do_scale = True
num_epoch = 1000
name = "gravity-150"
actor_policy_ckpt_list = [30, 80, 120, 300, 400]