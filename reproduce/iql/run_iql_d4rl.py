import torch
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer import D4RLTransitionBuffer
from offlinerllib.module.actor import ClippedGaussianActor, SquashedDeterministicActor
from offlinerllib.module.critic import Critic, DoubleCritic
from offlinerllib.module.net.mlp import MLP
from offlinerllib.policy.model_free import IQLPolicy
from offlinerllib.env.d4rl import get_d4rl_dataset
from offlinerllib.utils.eval import eval_offline_policy

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_dir=f"./log/iql/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
}, activate=not args.debug)
logger.log_config(args)
setup(args, logger)  # register args, logger, seed and device

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

offline_buffer = D4RLTransitionBuffer(dataset)

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

    
critic_q = DoubleCritic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape + action_shape, 
    hidden_dims=args.hidden_dims, 
).to(args.device)

critic_v = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    hidden_dims=args.hidden_dims, 
).to(args.device)

policy = IQLPolicy(
    actor=actor, critic_q=critic_q, critic_v=critic_v, 
    expectile=args.expectile, temperature=args.temperature, 
    tau=args.tau, 
    discount=args.discount, 
    max_action=args.max_action, 
    device=args.device, 
).to(args.device)
actor_opt_scheduler_steps = args.max_epoch * args.step_per_epoch if args.actor_opt_decay_schedule == "cosine" else None
policy.configure_optimizers(
    actor_lr=args.actor_lr, 
    critic_v_lr=args.critic_v_lr, 
    critic_q_lr=args.critic_q_lr, 
    actor_opt_scheduler_steps=actor_opt_scheduler_steps
)


# main loop
policy.train()
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in range(args.step_per_epoch):
        batch = offline_buffer.random_batch(args.batch_size)
        train_metrics = policy.update(batch)
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_offline_policy(env, policy, args.eval_episode, seed=args.seed)
    
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)

    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/iql/{args.name}/{args.task}/seed{args.seed}/policy/")
    
        