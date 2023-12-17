import gym
import numpy as np
import torch
import neorl
import copy
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

def neorl_normalize_reward(dataset):
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
    dataset["rewards"] /= max(max(returns), 1000)
    dataset["rewards"] *= 1000
    return dataset, {}

def neorl_normalize_obs(dataset):
    all_obs = np.concatenate([dataset["observations"], dataset["next_observations"]], axis=0)
    # all_obs = dataset["observations"]
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0)+1e-3
    dataset["observations"] = (dataset["observations"] - obs_mean) / obs_std
    dataset["next_observations"] = (dataset["next_observations"] - obs_mean) / obs_std
    return dataset, {
        "obs_mean": obs_mean, 
        "obs_std": obs_std
    }

def qlearning_dataset(env, dataset):
    N = dataset["reward"].shape[0]
    ends = copy.deepcopy(dataset["done"])
    terminals = copy.deepcopy(dataset["done"])
    # episode_step = 0
    # for i in range(N):
        # episode_step += 1
    
    return {
        "observations": dataset["obs"], 
        "actions": dataset["action"], 
        "next_observations": dataset["next_obs"], 
        "rewards": np.squeeze(dataset["reward"], axis=-1), 
        "terminals": np.squeeze(terminals, axis=-1), 
        "ends": np.squeeze(ends, axis=-1)
    }
        

def get_neorl_dataset(
    task, 
    normalize_reward=False, 
    normalize_obs=False, 
):
    domain, quality, version = task.split("-")
    env = neorl.make("-".join([domain, version]))
    env.reset()
    dataset = env.get_dataset(data_type=quality, train_num=1000)[0]
    dataset = qlearning_dataset(env, dataset)

    # define the normalized score function
    if domain == "HalfCheetah":
        min_score, max_score = -298, 12284
    elif domain == "Hopper":
        min_score, max_score = 5, 3294
    elif domain == "Walker2d":
        min_score, max_score = 1, 5143
    env.get_normalized_score = lambda x: (x-min_score) / (max_score - min_score)
    
    if normalize_reward:
        dataset, info = neorl_normalize_reward(dataset)
    if normalize_obs:
        dataset, info = neorl_normalize_obs(dataset)
        from gym.wrappers.transform_observation import TransformObservation
        env = TransformObservation(env, lambda obs: (obs-info["obs_mean"])/info["obs_std"])
    return env, dataset

