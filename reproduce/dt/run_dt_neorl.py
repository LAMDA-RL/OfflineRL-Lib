import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer import D4RLTrajectoryBuffer
from offlinerllib.module.net.attention.dt import DecisionTransformer
from offlinerllib.policy.model_free import DecisionTransformerPolicy
from offlinerllib.env.neorl_mujoco import get_neorl_dataset
from offlinerllib.utils.eval import eval_decision_transformer

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_dir=f"./log/dt/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_neorl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

offline_buffer = D4RLTrajectoryBuffer(dataset, seq_len=args.seq_len, return_scale=args.return_scale)

dt = DecisionTransformer(
    obs_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=args.seq_len+args.episode_len \
        if args.use_abs_timestep else args.seq_len, # this is for positional encoding
    num_heads=args.num_heads, 
    attention_dropout=args.attention_dropout, 
    residual_dropout=args.residual_dropout, 
    embed_dropout=args.embed_dropout, 
    pos_encoding=args.pos_encoding
).to(args.device)

policy = DecisionTransformerPolicy(
    dt=dt, 
    state_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    use_abs_timestep=args.use_abs_timestep, 
    policy_type=args.policy_type, 
    device=args.device
).to(args.device)
policy.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay, betas=args.betas, warmup_steps=args.warmup_steps)


# main loop
policy.train()
trainloader = DataLoader(
    offline_buffer, 
    batch_size=args.batch_size, 
    pin_memory=True,
    num_workers=args.num_workers
)
trainloader_iter = iter(trainloader)
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in range(args.step_per_epoch):
        batch = next(trainloader_iter)
        train_metrics = policy.update(batch, clip_grad=args.clip_grad)
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_decision_transformer(env, policy, args.target_returns, args.return_scale, args.eval_episode, seed=args.seed)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        
    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/dt/d4rl/{args.name}/{args.task}/seed{args.seed}/policy/")
    
    