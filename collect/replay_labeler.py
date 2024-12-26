import gym
import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
from robosuite.controllers import load_controller_config
import numpy as np
import torch
import wandb
import logging
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from UtilsRL.rl.buffer import TransitionSimpleReplay
from UtilsRL.env import make_dmc
from UtilsRL.env.wrapper import MujocoParamOverWrite

from offlinerllib.module.actor import SquashedGaussianActor
from offlinerllib.module.critic import Critic
from offlinerllib.module.net.mlp import MLP
from offlinerllib.policy.model_free import SACVPolicy
from offlinerllib.utils.eval import eval_online_policy
from offlinerllib.utils.gym_wrapper import GymWrapper


controller_config = load_controller_config(default_controller="OSC_POSE")
robosuite_env_args = {
    "horizon": 500,
    "controller_configs": controller_config,
    "use_object_obs": True,
    "reward_shaping": True,
    "hard_reset": False,
}

args = parse_args()
if args.env_type == "dmc":
    args.env = "-".join([args.domain.title(), args.task.title(), "v1"])
elif args.env_type == "mujoco":
    args.env = args.task
elif args.env_type == "robosuite":
    args.env = args.task
    args.robots = args.robots
exp_name = "_".join([args.env, "seed" + str(args.seed)])
logger = CompositeLogger(
    log_dir=f"./log/collect/{args.name}",
    name=exp_name,
    logger_config={
        "TensorboardLogger": {},
        # "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
    },
    activate=not args.debug,
)
setup(args, logger)

if args.env_type == "dmc":
    env = make_dmc(domain_name=args.domain, task_name=args.task)
    eval_env = make_dmc(domain_name=args.domain, task_name=args.task)
else:
    based_env = gym.make(args.env)
    based_eval_env = gym.make(args.env)
    env = MujocoParamOverWrite(
        based_env, overwrite_args=args.overwrite_args, do_scale=args.do_scale
    )
    eval_env = MujocoParamOverWrite(
        based_eval_env, overwrite_args=args.overwrite_args, do_scale=args.do_scale
    )

obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]


def get_policy(load_path):
    actor = SquashedGaussianActor(
        backend=torch.nn.Identity(),
        input_dim=obs_shape,
        output_dim=action_shape,
        conditioned_logstd=True,
        reparameterize=True,
        logstd_min=args.policy_logstd_min,
        logstd_max=args.policy_logstd_max,
        hidden_dims=args.actor_hidden_dims,
    ).to(args.device)

    critic_q = Critic(
        backend=torch.nn.Identity(),
        input_dim=obs_shape + action_shape,
        hidden_dims=args.critic_hidden_dims,
        ensemble_size=args.critic_q_num,
    ).to(args.device)

    critic_v = Critic(
        backend=torch.nn.Identity(),
        input_dim=obs_shape,
        hidden_dims=args.critic_hidden_dims,
        ensemble_size=1,
    ).to(args.device)

    policy = SACVPolicy(
        actor=actor,
        critic_q=critic_q,
        critic_v=critic_v,
        tau=args.tau,
        discount=args.discount,
        alpha=(-float(action_shape), args.alpha_lr) if args.auto_alpha else args.alpha,
        target_update_freq=args.target_update_freq,
        device=args.device,
    ).to(args.device)
    policy.configure_optimizers(args.actor_lr, args.critic_lr, args.critic_lr)
    policy.load_state_dict(torch.load(load_path))
    return policy


critic_policy = get_policy(
    f"./out/collect/{args.name}/{args.env}/seed{args.seed}/policy/policy_1000.pt"
)


# create folder if not exist
import os
import matplotlib.pyplot as plt


saved_path = f"./datasets/rpl/{args.name}/{args.env}/"
os.makedirs(saved_path, exist_ok=True)
raw_data = np.load(saved_path + "replay.npz")
processed_data = {}
for k in raw_data.keys():
    processed_data[k] = raw_data[k]
with torch.no_grad():
    obs = torch.tensor(raw_data["obs"], device=args.device).float()
    action = torch.tensor(raw_data["action"], device=args.device).float()
    next_obs = torch.tensor(raw_data["next_obs"], device=args.device).float()
    processed_data["q_value"] = critic_policy.critic_q(obs, action).mean(dim=0).cpu().numpy()
    processed_data["v_value"] = critic_policy.critic_v(obs).cpu().numpy()
    processed_data["next_v_value"] = critic_policy.critic_v(next_obs).cpu().numpy()
np.savez_compressed(saved_path + "labeled_replay.npz", **processed_data)
