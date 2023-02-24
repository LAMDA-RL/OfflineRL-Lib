import os
import torch
import wandb
from tqdm import trange
from offlinerllib.utils.d4rl import get_d4rl_dataset

from offlinerllib.module.net.attention import DecisionTransformer
from offlinerllib.policy.model_free import DecisionTransformerPolicy
from offlinerllib.utils.eval import eval_policy

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/dt/offline/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

dt = DecisionTransformer(
    obs_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    n_head=args.n_head, 
    attention_dropout=args.attention_dropout, 
    residual_dropout=args.residual_dropout, 
    embed_dropout=args.embed_dropout
).to(args.device)
dt_optim = torch.optim.AdamW(dt.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
dt_scheduler = torch.optim.lr_scheduler.LambdaLR(dt_optim, lambda step: min((step+1)/args.warmup_steps), 1)

policy = DecisionTransformerPolicy(
    dt=dt, 
    dt_optim=dt_optim, 
    state_dim=obs_shape, 
    action_dim=action_shape, 
    device=args.device
).to(args.device)


# main loop
