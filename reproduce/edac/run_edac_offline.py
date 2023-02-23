import os
import torch
import wandb
from tqdm import trange
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.policy.model_free import EDACPolicy

from offlinerllib.utils.eval import eval_offline_policy

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from offlinerllib.module.net.mlp import MLP
from offlinerllib.module.actor import SquashedGaussianActor
from offlinerllib.module.critic import Critic

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/edac/offline/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims)
actor = SquashedGaussianActor(
    backend=actor_backend, 
    input_dim=args.hidden_dims[-1], 
    output_dim=action_shape, 
    logstd_min=args.policy_logstd_min
).to(args.device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

critic = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape+action_shape, 
    hidden_dims=args.hidden_dims, 
    ensemble_size=args.num_critics
).to(args.device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

policy = EDACPolicy(
    actor=actor, 
    critic=critic, 
    actor_optim=actor_optim, 
    critic_optim=critic_optim, 
    tau=args.tau, 
    eta=args.eta, 
    gamma=args.gamma, 
    alpha=(-float(action_shape), args.alpha_lr) if args.auto_alpha else args.alpha, 
    do_reverse_update=args.do_reverse_update, 
    device=args.device
).to(args.device)


# main loop
policy.train()
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in trange(args.step_per_epoch):
        batch = dataset.sample(args.batch_size)
        train_metrics = policy.update(batch)
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_offline_policy(env, policy, args.eval_episode, seed=args.seed)
    
        logger.info(f"Epicode {i_epoch}: \n{eval_metrics}")

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)

    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/edac/offline/{args.name}/{args.task}/seed{args.seed}/policy/")
    
        