import torch
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer import D4RLTransitionBuffer
from offlinerllib.module.actor import ClippedGaussianActor
from offlinerllib.module.critic import Critic
from offlinerllib.module.net.mlp import MLP
from offlinerllib.policy.model_free import InACPolicy
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.utils.eval import eval_offline_policy

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/inac/offline/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=args.discard_last)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

offline_buffer = D4RLTransitionBuffer(dataset)

actor = ClippedGaussianActor(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    output_dim=action_shape, 
    reparameterize=True, 
    conditioned_logstd=False, 
    logstd_min=-6, 
    logstd_max=0,
    hidden_dims=args.hidden_dims, 
    device=args.device
).to(args.device)
behavior = ClippedGaussianActor(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    output_dim=action_shape, 
    reparameterize=True, 
    conditioned_logstd=False, 
    logstd_min=-6, 
    logstd_max=0, 
    hidden_dims=args.hidden_dims, 
    device=args.device
).to(args.device)

critic_q = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape+action_shape, 
    hidden_dims=args.hidden_dims, 
    ensemble_size=2, 
    device=args.device
).to(args.device)

critic_v = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    hidden_dims=args.hidden_dims, 
    device=args.device
).to(args.device)

actor_optim = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
behavior_optim = torch.optim.Adam(behavior.parameters(), lr=args.learning_rate)
critic_q_optim = torch.optim.Adam(critic_q.parameters(), lr=args.learning_rate)
critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.learning_rate)

policy = InACPolicy(
    actor=actor, behavior=behavior, critic_q=critic_q, critic_v=critic_v, 
    actor_optim=actor_optim, 
    behavior_optim=behavior_optim,
    critic_q_optim=critic_q_optim, 
    critic_v_optim=critic_v_optim, 
    temperature=args.temperature, 
    discount=args.discount, 
    tau=args.tau, 
    device=args.device
).to(args.device)

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
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/inac/offline/{args.name}/{args.task}/seed{args.seed}/policy/")

