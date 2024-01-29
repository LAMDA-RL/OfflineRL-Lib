import gym
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
from offlinerllib.policy.model_free import SACPolicy
from offlinerllib.utils.eval import eval_online_policy


args = parse_args()
if args.env_type == "dmc":
    args.env = "-".join([args.domain.title(), args.task.title(), "v1"])
elif args.env_type == "mujoco":
    args.env = args.task
exp_name = "_".join([args.env, "seed" + str(args.seed)])
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
setup(args, logger)

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

    policy = SACPolicy(
        actor=actor,
        critic=critic_q,
        tau=args.tau,
        discount=args.discount,
        alpha=(-float(action_shape), args.alpha_lr) if args.auto_alpha else args.alpha,
        target_update_freq=args.target_update_freq,
        device=args.device,
    ).to(args.device)
    policy.configure_optimizers(args.actor_lr, args.critic_lr)
    policy.load_state_dict(torch.load(load_path))
    return policy


actor_policy = get_policy(
    f"../out/sac-08/{args.name}/{args.env}/seed{args.seed}/policy/policy_250.pt"
)
critic_policy = get_policy(
    f"../out/sac-08/{args.name}/{args.env}/seed{args.seed}/policy/policy_3000.pt"
)


buffer = TransitionSimpleReplay(
    max_size=args.max_trajectory_length,
    field_specs={
        "observations": {
            "shape": [
                obs_shape,
            ],
            "dtype": np.float32,
        },
        "actions": {
            "shape": [
                action_shape,
            ],
            "dtype": np.float32,
        },
        "next_observations": {
            "shape": [
                obs_shape,
            ],
            "dtype": np.float32,
        },
        "rewards": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "terminals": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "timeouts": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "masks": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "q_values": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "v_values": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
    },
)


# main loop
num_epoch = 10000
collected_data = []
obs, terminal = env.reset(), False
cur_traj_length = cur_traj_return = 0
for i_epoch in trange(1, num_epoch + 1):
    buffer.reset()
    for i_step in range(1, args.max_trajectory_length + 1):
        action = actor_policy.select_action(obs)

        next_obs, reward, done, info = env.step(action)
        cur_traj_length += 1
        cur_traj_return += reward
        
        timeout = False
        terminal = False
        if i_epoch == args.max_trajectory_length:
            timeout = True
        elif done:
            terminal = True

        with torch.no_grad():
            buffer.add_sample(
                {
                    "observations": obs,
                    "actions": action,
                    "next_observations": next_obs,
                    "rewards": reward,
                    "terminals": terminal,
                    "timeouts": timeout,
                    "masks": 1.0,
                    "q_values": (critic_policy.critic(torch.tensor(obs, device=args["device"]).float(), torch.tensor(action, device=args["device"]).float()).mean(dim=0)).cpu().numpy(),
                    "v_values": (critic_policy.critic(torch.tensor(obs, device=args["device"]).float(), torch.tensor(critic_policy.select_action(obs), device=args["device"]).float()).mean(dim=0)).cpu().numpy(),
                }
            )
        obs = next_obs
        if terminal:
            break
    collected_data.append(
        {
            **buffer.fields,
            "episode_return": cur_traj_return,
            "episode_length": cur_traj_length,
        }
    )
    obs = env.reset()
    cur_traj_length = cur_traj_return = 0


processed_data = { k: [] for k in collected_data[0].keys() }
for data in collected_data:
    for k, v in data.items():
        processed_data[k].append(v)
assert all([len(v) == len(processed_data[list(processed_data.keys())[0]]) for k, v in processed_data.items()])
processed_data = { k: np.array(v) for k, v in processed_data.items() }


# save processed_data to f"../datasets/sac-08/{args.name}/{args.env}/seed{args.seed}/data.npz"
# create folder if not exist
import os
os.makedirs(f"../datasets/sac-08/{args.name}/{args.env}/seed{args.seed}", exist_ok=True)
np.savez_compressed(
    f"../datasets/sac-08/{args.name}/{args.env}/seed{args.seed}/data.npz",
    **processed_data
)