import numpy as np

import gym
import d4rl

from offlinerllib.buffer.dataset import D4RLDataset

def _calc_terminal(dataset):
    terminal = dataset["terminals"].copy()
    count = 0
    for i in range(len(terminal) - 1):
        if terminal[i]:
            continue
        elif np.linalg.norm(dataset["observations"][i+1] - dataset["next_observations"][i]) > 1e-6:
            terminal[i] = True
            count += 1
    # terminal[-1] = True
    return terminal
    
def _antmaze_normalize_reward(dataset):
    dataset["rewards"] -= 1.0
    return dataset, {}
    
def _normalize_reward(dataset):
    terminal = _calc_terminal(dataset)
    reward = dataset["rewards"]
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(reward, terminal):
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
        
def get_d4rl_dataset(task, normalize_reward=False, normalize_obs=False):
    env = gym.make(task)
    dataset = d4rl.qlearning_dataset(env)
    if normalize_reward:
        if "antmaze" in task:
            dataset, _ = _antmaze_normalize_reward(dataset)
        elif "halfcheetah" in task or "hopper" in task or "walker2d" in task:
            dataset, _ = _normalize_reward(dataset)
    if normalize_obs:
        dataset, info = _normalize_obs(dataset)
        from gym.wrappers.transform_observation import TransformObservation
        env = TransformObservation(env, lambda obs: (obs - info["obs_mean"])/info["obs_std"])
    return env, D4RLDataset(dataset)
        