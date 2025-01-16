from collect.config.collect.base import *

task = "HalfCheetah-v3"
overwrite_args = {"gravity": 3.0}
do_scale = True
name = "gravity-300"
actor_policy_ckpt_list = [10, 20, 30, 100, 300]