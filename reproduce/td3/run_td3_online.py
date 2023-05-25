import gym
import numpy as np
import torch
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from UtilsRL.rl.buffer import TransitionSimpleReplay
from UtilsRL.env import make_dmc

from offlinerllib.module.actor import SquashedDeterministicActor
from offlinerllib.module.critic import Critic
from offlinerllib.policy.model_free import TD3Policy
from offlinerllib.utils.eval import eval_online_policy
from offlinerllib.utils.noise import WhiteNoise

args = parse_args()
if args.env_type == "dmc":
    args.env = "-".join([args.domain.title(), args.task.title(), "v1"])
elif args.env_type == "mujoco":
    args.env = args.task
exp_name = "_".join([args.env, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/td3/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

if args.env_type == "dmc":
    env = make_dmc(domain_name=args.domain, task_name=args.task)
    eval_env = make_dmc(domain_name=args.domain, task_name=args.task)
else:
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]
    
actor = SquashedDeterministicActor(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    output_dim=action_shape, 
    hidden_dims=args.hidden_dims
).to(args.device)

critic = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape+action_shape, 
    hidden_dims=args.hidden_dims, 
    ensemble_size=2    
).to(args.device)

policy = TD3Policy(
    actor=actor, 
    critic=critic, 
    actor_update_interval=args.actor_update_interval, 
    policy_noise=args.policy_noise, 
    noise_clip=args.noise_clip, 
    exploration_noise=WhiteNoise(mu=0, sigma=args.exploration_noise), 
    device=args.device
).to(args.device)
policy.configure_optimizers(args.actor_lr, args.critic_lr)

buffer = TransitionSimpleReplay(
    max_size=args.max_buffer_size, 
    field_specs={
        "observations": {"shape": [obs_shape, ], "dtype": np.float32}, 
        "actions": {"shape": [action_shape, ], "dtype": np.float32}, 
        "next_observations": {"shape": [obs_shape, ], "dtype": np.float32}, 
        "rewards": {"shape": [1, ], "dtype": np.float32, }, 
        "terminals": {"shape": [1, ], "dtype": np.float32, }, 
    }
)
buffer.reset()

# main loop
obs, terminal = env.reset(), False
cur_traj_length = cur_traj_return = 0
all_traj_lengths = [0]
all_traj_returns = [0]
for i_epoch in trange(1, args.num_epoch+1):
    for i_step in range(args.step_per_epoch):
        if i_epoch < args.random_policy_epoch+1:
            action = env.action_space.sample()
        else:
            action = policy.select_action(obs, deterministic=False)
        
        next_obs, reward, terminal, info = env.step(action)
        cur_traj_length += 1
        cur_traj_return += reward
        if cur_traj_length >= args.max_trajectory_length:
            terminal = False
        buffer.add_sample({
            "observations": obs, 
            "actions": action, 
            "next_observations": next_obs, 
            "rewards": reward, 
            "terminals": terminal, 
        })
        obs = next_obs
        if terminal or cur_traj_length >= args.max_trajectory_length:
            obs = env.reset()
            all_traj_returns.append(cur_traj_return)
            all_traj_lengths.append(cur_traj_length)
            cur_traj_length = cur_traj_return = 0
        
        batch_data = buffer.random_batch(args.batch_size)
        train_metrics = policy.update(batch_data)
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_online_policy(eval_env, policy, args.eval_episode, seed=args.seed)
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
        logger.log_scalars("rollout", {
            "episode_return": all_traj_returns[-1], 
            "episode_length": all_traj_lengths[-1]
        }, step=i_epoch)
    
    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/td3/{args.name}/{args.env}/seed{args.seed}/policy/")