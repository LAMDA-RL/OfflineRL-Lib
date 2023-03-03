import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer import D4RLTrajectoryBuffer
from offlinerllib.module.net.attention import DecisionTransformer
from offlinerllib.policy.model_free import DecisionTransformerPolicy
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.utils.eval import eval_decision_transformer

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/dt/offline/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

offline_buffer = D4RLTrajectoryBuffer(dataset, seq_len=args.seq_len, return_scale=args.return_scale)

dt = DecisionTransformer(
    obs_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    num_heads=args.num_heads, 
    attention_dropout=args.attention_dropout, 
    residual_dropout=args.residual_dropout, 
    embed_dropout=args.embed_dropout
).to(args.device)
dt_optim = torch.optim.AdamW(dt.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
dt_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(dt_optim, lambda step: min((step+1)/args.warmup_steps, 1))

policy = DecisionTransformerPolicy(
    dt=dt, 
    dt_optim=dt_optim, 
    state_dim=obs_shape, 
    action_dim=action_shape, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    device=args.device
).to(args.device)


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
        dt_optim_scheduler.step()
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_decision_transformer(env, policy, args.target_returns, args.return_scale, args.eval_episode, seed=args.seed)
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
        logger.log_scalar("misc/learning_rate", dt_optim_scheduler.get_last_lr()[0], step=i_epoch)
        
    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/dt/offline/{args.name}/{args.task}/seed{args.seed}/policy/")
    
    