import torch
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer import D4RLTransitionBuffer
from offlinerllib.module.actor import ClippedGaussianActor
from offlinerllib.module.critic import Critic
from offlinerllib.module.net.mlp import MLP
from offlinerllib.policy.model_free import AWACPolicy
from offlinerllib.env.mixed import get_mixed_d4rl_mujoco_datasets
from offlinerllib.utils.eval import eval_offline_policy

args = parse_args()
args.task = "-".join(["mixed", args.agent, args.quality1, args.quality2, str(args.ratio)])
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_dir=f"./log/awac/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
}, activate=not args.debug)
setup(args, logger)

env, dataset = get_mixed_d4rl_mujoco_datasets(
    agent=args.agent, 
    quality1=args.quality1, 
    quality2=args.quality2, 
    N=args.num_data, 
    ratio=args.ratio, 
    keep_traj=args.keep_traj, 
    normalize_obs=args.normalize_obs, 
    normalize_reward=args.normalize_reward
)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

offline_buffer = D4RLTransitionBuffer(dataset)

actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims)
actor = ClippedGaussianActor(
    backend=actor_backend, 
    input_dim=args.hidden_dims[-1], 
    output_dim=action_shape, 
).to(args.device)

critic = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape+action_shape, 
    hidden_dims=args.hidden_dims, 
    ensemble_size=2
).to(args.device)

policy = AWACPolicy(
    actor=actor, 
    critic=critic, 
    aw_lambda=args.aw_lambda, 
    tau=args.tau, 
    discount=args.discount, 
    device=args.device
).to(args.device)
policy.configure_optimizers(args.actor_lr, args.critic_lr)


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
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/awac/{args.name}/{args.task}/seed{args.seed}/policy/")