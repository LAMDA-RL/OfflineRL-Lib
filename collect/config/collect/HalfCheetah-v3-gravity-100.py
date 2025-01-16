from collect.config.collect.base import *

task = "HalfCheetah-v3"
overwrite_args = {"gravity": 1.0}
do_scale = True
name = "gravity-100"
actor_policy_ckpt_list = [50, 80, 120, 250, 450]