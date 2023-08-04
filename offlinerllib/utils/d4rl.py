import gym
import numpy as np
import d4rl
import torch

from offlinerllib.utils.terminal import get_termination_fn

def antmaze_normalize_reward(dataset):
    dataset["rewards"] -= 1.0
    return dataset, {}
    
def mujoco_normalize_reward(dataset):
    split_points = dataset["ends"].copy()
    split_points[-1] = False   # the last traj may be incomplete, so we discard them
    reward = dataset["rewards"]
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(reward, split_points):
        ep_ret += float(r)
        ep_len += 1
        if d:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    dataset["rewards"] /= max(returns) - min(returns)
    dataset["rewards"] *= 1000
    return dataset, {}
    
def _normalize_obs(dataset):
    all_obs = np.concatenate([dataset["observations"], dataset["next_observations"]], axis=0)
    # all_obs = dataset["observations"]
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0)+1e-3
    dataset["observations"] = (dataset["observations"] - obs_mean) / obs_std
    dataset["next_observations"] = (dataset["next_observations"] - obs_mean) / obs_std
    return dataset, {
        "obs_mean": obs_mean, 
        "obs_std": obs_std
    }

def qlearning_dataset(env, dataset=None, terminate_on_end: bool=False, discard_last: bool=True, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    end_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)  # Thus, the next_obs for the last timestep is totally false
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        end = False
        episode_step += 1

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if final_timestep:
            if not done_bool:
                if not terminate_on_end:
                    if discard_last:
                        episode_step = 0
                        end_[-1] = True
                        continue
                else: 
                    done_bool = True
        if final_timestep or done_bool:
            end = True
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        end_.append(end)
    
    end_[-1] = True   # the last traj will be ended whatsoever
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        "ends": np.array(end_)
    }
    
        
def get_d4rl_dataset(task, normalize_reward=False, normalize_obs=False, terminate_on_end: bool=False, discard_last: bool=True, return_termination_fn=False, **kwargs):
    env = gym.make(task)
    dataset = qlearning_dataset(env, terminate_on_end=terminate_on_end, discard_last=discard_last, **kwargs)
    if normalize_reward:
        if "antmaze" in task:
            dataset, _ = antmaze_normalize_reward(dataset)
        elif "halfcheetah" in task or "hopper" in task or "walker2d" in task or "ant" in task:
            dataset, _ = mujoco_normalize_reward(dataset)
    termination_fn = get_termination_fn(task)
    if normalize_obs:
        dataset, info = _normalize_obs(dataset)
        from gym.wrappers.transform_observation import TransformObservation
        env = TransformObservation(env, lambda obs: (obs - info["obs_mean"])/info["obs_std"])
        termination_fn = get_termination_fn(task, info["obs_mean"], info["obs_std"])
    if return_termination_fn:
        return env, dataset, termination_fn
    else:
        return env, dataset
        

# below is for dataset generation
@torch.no_grad()
def gen_d4rl_dataset(task, policy, num_data, policy_is_online=False, random=False, normalize_obs: bool=False, seed=0, **d4rl_kwargs):
    if not hasattr(policy, "actor"):
        raise AttributeError("Policy does not have actor member")
    if policy_is_online:
        env = gym.make(task)
        transform_fn = lambda obs: obs
    else:
        env = gym.make(task)
        dataset = qlearning_dataset(env, **d4rl_kwargs)
        if normalize_obs:
            dataset, info = _normalize_obs(dataset)
            transform_fn = lambda obs: (obs - info["obs_mean"]) / (info["obs_std"] + 1e-3)
        else:
            transform_fn = lambda obs: obs
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    def init_dict():
        return {
            "observations": [], 
            "actions": [], 
            "next_observations": [], 
            "rewards": [], 
            "terminals": [], 
            "timeouts": [], 
            "infos/action_log_probs": [], 
            "infos/qpos": [], 
            "infos/qvel": []
        }
    
    data = init_dict()
    traj_data = init_dict()
    
    obs, done, return_, length = env.reset(seed=seed), 0, 0, 0
    while len(data["rewards"]) < num_data:
        if random:
            action = env.action_space.sample()
            logprob = np.log(1.0 / np.prod(env.action_space.high - env.action_space.low))
        else:
            obs_torch = torch.from_numpy(transform_fn(obs)).float().to(policy.device)
            action, logprob, *_ = policy.actor.sample(obs_torch, determinisitc=False)
            action = action.squeeze().cpu().numpy()
            logprob = logprob.squeeze().cpu().numpy()
        # mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()
        ns, rew, done, infos = env.step(action)
        return_ += rew
        
        length += 1
        timeout = False
        terminal = False
        
        if length == env._max_episode_steps:
            timeout = True
        elif done:
            terminal = True
            
        for _key, _value in {
            "observations": obs, 
            "actions": action, 
            "next_observations": ns, 
            "rewards": rew, 
            "terminals": terminal, 
            "timeouts": timeout, 
            "infos/action_log_probs": logprob, 
            "infos/qpos": qpos, 
            "infos/qvel": qvel
        }.items():
            traj_data[_key].append(_value)
        obs = ns
        if terminal or timeout:
            print(f"finished trajectory, len={length}, return={return_}")
            s = env.reset()
            length = return_ = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = init_dict()
            
    new_data = {_key: np.asarray(_value).astype(np.float32) for _key, _value in data.items()}
    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data

    