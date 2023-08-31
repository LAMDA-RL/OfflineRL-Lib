import gym
import numpy as np
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from UtilsRL.env import make_dmc

from offlinerllib.module.critic import Critic
from offlinerllib.utils.eval import eval_online_policy
from offlinerllib.utils.noise import WhiteNoise
from offlinerllib.buffer.lap_buffer import LAPBuffer
from offlinerllib.module.td7_net import TD7Encoder, TD7Actor, TD7Critic
from offlinerllib.policy.model_free.td7 import TD7Policy

args = parse_args()
if args.env_type == "dmc":
    args.env = "-".join([args.domain.title(), args.task.title(), "v1"])
elif args.env_type == "mujoco":
    args.env = args.task
exp_name = "_".join([args.env, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_dir=f"./log/td7/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
}, activate=not args.debug)
setup(args, logger)

if args.env_type == "dmc":
    env = make_dmc(domain_name=args.domain, task_name=args.task)
    eval_env = make_dmc(domain_name=args.domain, task_name=args.task)
else:
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

actor = TD7Actor(
    state_dim=obs_shape, 
    action_dim=action_shape, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim, 
).to(args.device)
critic = TD7Critic(
    state_dim=obs_shape, 
    action_dim=action_shape, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim, 
    critic_num=2, 
)
encoder = TD7Encoder(
    state_dim=obs_shape, 
    action_dim=action_shape, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim
).to(args.device)

policy = TD7Policy(
    actor=actor, 
    critic=critic, 
    encoder=encoder, 
    offline=False, 
    actor_update_interval=args.actor_update_interval, 
    target_update_interval=args.target_update_interval, 
    policy_noise=args.policy_noise, 
    noise_clip=args.noise_clip, 
    exploration_noise=WhiteNoise(mu=0, sigma=args.exploration_noise), 
    lam=0.0, 
    discount=args.discount, 
    max_action=args.max_action, 
    device=args.device
).to(args.device)
policy.configure_optimizers(args.actor_lr, args.critic_lr, args.encoder_lr)

buffer = LAPBuffer(
    max_size=args.max_buffer_size, 
    field_specs={
        "observations": {"shape": [obs_shape, ], "dtype": np.float32}, 
        "actions": {"shape": [action_shape, ], "dtype": np.float32}, 
        "next_observations": {"shape": [obs_shape, ], "dtype": np.float32}, 
        "rewards": {"shape": [1, ], "dtype": np.float32, }, 
        "terminals": {"shape": [1, ], "dtype": np.float32, }, 
    }, 
    prioritized=True, 
    alpha=0.4, 
    min_priority=1.0
)


# main loop
obs, terminal = env.reset(), False
cur_traj_length = cur_traj_return = 0
all_traj_lengths = [0]
all_traj_returns = [0]
allow_train = False
for i_epoch in trange(1, args.num_epoch+1):
    for i_step in range(args.step_per_epoch):
        if not allow_train:
            action = env.action_space.sample()
        else:
            action = policy.select_action(obs, deterministic=False)
        next_obs, reward, ep_finished, info = env.step(action)
        cur_traj_length += 1
        cur_traj_return += reward
        terminal = ep_finished if cur_traj_length < args.max_trajectory_length else False
        buffer.add_sample({
            "observations": obs, 
            "actions": action, 
            "next_observations": next_obs, 
            "rewards": reward, 
            "terminals": terminal, 
        })
        obs = next_obs

        if allow_train and not args.use_checkpoint: 
            batch, batch_idx = buffer.random_batch(args.batch_size, return_idx=True)
            train_metrics, new_td_error = policy.update(batch)
            buffer.batch_update(batch_idx, new_td_error)
        else:
            train_metrics = {}
        
        if ep_finished:
            obs = env.reset()
            all_traj_returns.append(cur_traj_return)
            all_traj_lengths.append(cur_traj_length)
            cur_traj_length = cur_traj_return = 0
            buffer.reset_max_priority()
            
            if (i_epoch-1) * args.step_per_epoch + i_step >= args.step_before_training:
                allow_train = True
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_online_policy(eval_env, policy, args.eval_episode, seed=args.seed)
        logger.info(f"Epoch {i_epoch}: \n{eval_metrics}")
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
        logger.log_scalars("rollout", {
            "episode_return": all_traj_returns[-1], 
            "episode_length": all_traj_lengths[-1]
        }, step=i_epoch)

    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/td7/{args.name}/{args.env}/seed{args.seed}/policy/")