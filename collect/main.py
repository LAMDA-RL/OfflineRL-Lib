import os
import gym
import numpy as np
import torch
import wandb
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

buffer = TransitionSimpleReplay(
    max_size=args.max_buffer_size,
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
    },
)
buffer.reset()

traj_buffer = TransitionSimpleReplay(
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
    },
)

# main loop
obs, terminal = env.reset(), False
traj_buffer.reset()
cur_traj_length = cur_traj_return = 0
collected_data = []
all_traj_lengths = [0]
all_traj_returns = [0]
for i_epoch in trange(1, args.num_epoch + 1):
    for i_step in range(args.step_per_epoch):
        if i_epoch < args.random_policy_epoch + 1:
            action = env.action_space.sample()
        else:
            action = policy.select_action(obs)

        next_obs, reward, done, info = env.step(action)
        cur_traj_length += 1
        cur_traj_return += reward
        timeout = False
        terminal = False
        if i_step == args.max_trajectory_length - 1:
            timeout = True
        elif done:
            terminal = True
        buffer.add_sample(
            {
                "observations": obs,
                "actions": action,
                "next_observations": next_obs,
                "rewards": reward,
                "terminals": done,
            }
        )
        traj_buffer.add_sample(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "reward": reward,
                "terminal": terminal,
                "timeout": timeout,
                "mask": 1.0,
            }
        )
        obs = next_obs
        if terminal or timeout or cur_traj_length >= args.max_trajectory_length:
            obs = env.reset()
            all_traj_returns.append(cur_traj_return)
            all_traj_lengths.append(cur_traj_length)
            collected_data.append(
                {
                    **traj_buffer.fields,
                    "episode_return": cur_traj_return,
                    "episode_length": cur_traj_length,
                }
            )
            cur_traj_length = cur_traj_return = 0
            traj_buffer.reset()

        if i_epoch < args.warmup_epoch + 1:
            train_metrics = {}
        else:
            batch_data = buffer.random_batch(args.batch_size)
            train_metrics = policy.update(batch_data)

    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_online_policy(
            eval_env, policy, args.eval_episode, seed=args.seed
        )
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
        logger.log_scalars(
            "rollout",
            {
                "episode_return": all_traj_returns[-1],
                "episode_length": all_traj_lengths[-1],
            },
            step=i_epoch,
        )

    if i_epoch % args.save_interval == 0:
        logger.log_object(
            name=f"policy_{i_epoch}.pt",
            object=policy.state_dict(),
            path=f"./out/collect/{args.name}/{args.env}/seed{args.seed}/policy/",
        )

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
processed_data["mask"] = processed_data["mask"].squeeze(-1)
for k, v in processed_data.items():
    print(k, v.shape)

# create folder if not exist
import matplotlib.pyplot as plt

saved_path = f"./datasets/rpl/{args.env}/{args.name}/"
os.makedirs(saved_path, exist_ok=True)
np.savez_compressed(saved_path + "replay.npz", **processed_data)

plt.hist(processed_data['episode_return'], bins=100)
# title
plt.title(f'{args.name}/{args.env}')
# save to saved_path/episode_return.png
plt.savefig(saved_path + 'replay_episode_return.png')
