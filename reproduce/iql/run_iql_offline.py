import os
import torch
import wandb
from tqdm import trange
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.policy.model_free import IQLPolicy
from offlinerllib.utils.eval import eval_offline_policy

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from offlinerllib.module.net.mlp import MLP
from offlinerllib.module.actor import SquashedDeterministicActor, ClippedGaussianActor
from offlinerllib.module.critic import DoubleCritic, Critic

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/iql/offline/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims, dropout=args.dropout)
if args.iql_deterministic: 
    actor = SquashedDeterministicActor(
        backend=actor_backend, 
        input_dim=args.hidden_dims[-1], 
        output_dim=action_shape, 
    ).to(args.device)
else:
    actor = ClippedGaussianActor(
        backend=actor_backend, 
        input_dim=args.hidden_dims[-1], 
        output_dim=action_shape, 
        conditioned_logstd=args.conditioned_logstd, 
        logstd_min = args.policy_logstd_min
    ).to(args.device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
if args.actor_opt_decay_schedule:
    actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.max_epoch * args.step_per_epoch)
else:
    actor_lr_scheduler = None
    
critic_q = DoubleCritic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape + action_shape, 
    hidden_dims=args.hidden_dims, 
).to(args.device)
critic_q_optim = torch.optim.Adam(critic_q.parameters(), lr=args.critic_q_lr)

critic_v = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    hidden_dims=args.hidden_dims, 
).to(args.device)
critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)

policy = IQLPolicy(
    actor=actor, critic_q=critic_q, critic_v=critic_v, 
    actor_optim=actor_optim, critic_q_optim=critic_q_optim, critic_v_optim=critic_v_optim, 
    expectile=args.expectile, temperature=args.temperature, 
    tau=args.tau, 
    discount=args.discount, 
    max_action=args.max_action, 
    device=args.device, 
).to(args.device)


# main loop
policy.train()
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in range(args.step_per_epoch):
        batch = dataset.sample(args.batch_size)
        train_metrics = policy.update(batch)
        if actor_lr_scheduler is not None:
            actor_lr_scheduler.step()
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_offline_policy(env, policy, args.eval_episode, seed=args.seed)
    
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)

    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/iql/offline/{args.name}/{args.task}/seed{args.seed}/policy/")
    
        