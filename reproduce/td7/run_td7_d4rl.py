import numpy as np
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer.lap_buffer import LAPBuffer
from offlinerllib.module.td7_net import TD7Encoder, TD7Actor, TD7Critic
from offlinerllib.policy.model_free.td7 import TD7Policy
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.utils.eval import eval_offline_policy

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/td7/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
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
    offline=True, 
    actor_update_interval=args.actor_update_interval, 
    target_update_interval=args.target_update_interval, 
    policy_noise=args.policy_noise, 
    noise_clip=args.noise_clip, 
    exploration_noise=None, 
    lam=args.lam, 
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
buffer.add_sample({
    k: v[:, None] if k in {"rewards", "terminals"} else v for k, v in dataset.items() \
        if k in {"observations", "actions", "next_observations", "rewards", "terminals"}
})


# main loop
policy.train()
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in range(args.step_per_epoch):
        batch, batch_idx = buffer.random_batch(args.batch_size, return_idx=True)
        train_metrics, new_td_error = policy.update(batch)
        buffer.batch_update(batch_idx, new_td_error)
    
    buffer.reset_max_priority()
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_offline_policy(env, policy, args.eval_episode, seed=args.seed)
    
        logger.info(f"Epoch {i_epoch}: \n{eval_metrics}")

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)

    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/td7/{args.name}/{args.task}/seed{args.seed}/policy/")
    
