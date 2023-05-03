from offlinerllib.env.mixed import get_mixed_d4rl_mujoco_datasets
from offlinerllib.env.d4rl import get_d4rl_dataset


def get_env_and_dataset(args):
    if "mixed" in args.task:
        quality1, quality2 = args.qualities
        agent = args.task.split("-")[1]
        N = args.N
        ratio = args.ratio
        keep_traj = args.keep_traj
        normalize_obs = args.normalize_obs
        reward_norm = args.reward_norm
        return get_mixed_d4rl_mujoco_datasets(
            agent, quality1, quality2, N, ratio, keep_traj, normalize_obs, reward_norm
        )
    elif "branched" in args.task:
        raise NotImplementedError
    else:
        return get_d4rl_dataset(args.task, args.normalize_reward, args.normalize_obs)