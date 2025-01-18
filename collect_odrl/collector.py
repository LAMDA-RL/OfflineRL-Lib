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
from offlinerllib.env.odrl_envs.mujoco.call_mujoco_env import call_mujoco_env

args = parse_args()
assert args.env_type == "odrl"
args.env = args.task
exp_name = "_".join([args.env, "seed" + str(args.seed)])
logger = CompositeLogger(
    log_dir=f"./log/{args.name}/{args.env}-{args.shift_level}",
    name=exp_name,
    logger_config={
        "TensorboardLogger": {},
        # "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
    },
    activate=not args.debug,
)
setup(args, logger)

env = call_mujoco_env({"env_name": args.env, "shift_level": args.shift_level})
eval_env = call_mujoco_env({"env_name": args.env, "shift_level": args.shift_level})

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

actor_policy_ckpt_list = args.actor_policy_ckpt_list
critic_policy = get_policy(
    f"./out/{args.name}/{args.env}-{args.shift_level}/seed{args.seed}/policy/policy_3000.pt"
)


buffer = TransitionSimpleReplay(
    max_size=args.max_trajectory_length,
    field_specs={
        "obs": {
            "shape": [
                obs_shape,
            ],
            "dtype": np.float32,
        },
        "action": {
            "shape": [
                action_shape,
            ],
            "dtype": np.float32,
        },
        "next_obs": {
            "shape": [
                obs_shape,
            ],
            "dtype": np.float32,
        },
        "reward": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "terminal": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "timeout": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "mask": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "q_value": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "v_value": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "next_v_value": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
        "logprob": {
            "shape": [
                1,
            ],
            "dtype": np.float32,
        },
    },
)


# main loop
num_epoch = args.num_epoch
collected_data = []
obs, terminal = env.reset(), False
cur_traj_length = cur_traj_return = 0

def select_action_with_noise(
    policy: SACVPolicy,
    obs: np.ndarray,
    noise_std: float = 0.2,
) -> np.ndarray:
    action = policy.select_action(obs)
    noise = np.random.normal(0, noise_std, size=action.shape)
    action_noisy = action + noise
    # Optionally clip the action to the valid range
    action_noisy = np.clip(action_noisy, -1, 1)  # Adjust bounds if necessary for tanh
    return action_noisy

for i_epoch in trange(0, num_epoch):
    if i_epoch % (num_epoch // len(actor_policy_ckpt_list)) == 0:
        ckpt_index = min(i_epoch // (num_epoch // len(actor_policy_ckpt_list)), len(actor_policy_ckpt_list) - 1)
        actor_policy = get_policy(
            f"./out/{args.name}/{args.env}-{args.shift_level}/seed{args.seed}/policy/policy_{str(actor_policy_ckpt_list[ckpt_index])}.pt"
        )
    buffer.reset()
    for i_step in range(args.max_trajectory_length):
        action = select_action_with_noise(actor_policy, obs)

        next_obs, reward, done, info = env.step(action)
        cur_traj_length += 1
        cur_traj_return += reward

        timeout = False
        terminal = False
        if i_step == args.max_trajectory_length - 1:
            timeout = True
        elif done:
            terminal = True

        with torch.no_grad():
            buffer.add_sample(
                {
                    "obs": obs,
                    "action": action,
                    "next_obs": next_obs,
                    "reward": reward,
                    "terminal": terminal,
                    "timeout": timeout,
                    "mask": 1.0,
                    "q_value": (
                        critic_policy.critic_q(
                            torch.tensor(obs, device=args["device"]).float(),
                            torch.tensor(action, device=args["device"]).float(),
                        ).mean(dim=0)
                    )
                    .cpu()
                    .numpy(),
                    "v_value": (
                        critic_policy.critic_v(
                            torch.tensor(obs, device=args["device"]).float(),
                        )
                    )
                    .cpu()
                    .numpy(),
                    "next_v_value": (
                        critic_policy.critic_v(
                            torch.tensor(next_obs, device=args["device"]).float(),
                        )
                    )
                    .cpu()
                    .numpy(),
                    "logprob": (
                        critic_policy.actor.evaluate(
                            torch.tensor(obs, device=args["device"]).float(),
                            torch.tensor(action, device=args["device"]).float(),
                        )[0]
                    )
                    .cpu()
                    .numpy(),
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


processed_data = {k: [] for k in collected_data[0].keys()}
for data in collected_data:
    for k, v in data.items():
        processed_data[k].append(v)
assert all(
    [
        len(v) == len(processed_data[list(processed_data.keys())[0]])
        for k, v in processed_data.items()
    ]
)
processed_data = {k: np.array(v) for k, v in processed_data.items()}
for k, v in processed_data.items():
    print(k, v.shape)
processed_data["mask"] = processed_data["mask"].squeeze(-1)
for k, v in processed_data.items():
    print(k, v.shape)

# create folder if not exist
import os
import matplotlib.pyplot as plt

saved_path = f"./datasets/rpl/{args.name}/{args.env}-{args.shift_level}/"
os.makedirs(saved_path, exist_ok=True)
np.savez_compressed(saved_path + "data.npz", **processed_data)

plt.hist(processed_data['episode_return'], bins=100)
# title
plt.title(f'{args.env}-{args.shift_level}')
# save to saved_path/episode_return.png
plt.savefig(saved_path + 'episode_return.png')
