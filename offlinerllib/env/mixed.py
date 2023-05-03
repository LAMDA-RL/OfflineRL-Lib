import random
import gym
import numpy as np

from offlinerllib.env.d4rl import get_d4rl_dataset, _normalize_obs

def minmax_reward_normalizer(dataset):
    reward = dataset["rewards"]
    dataset["rewards"] = (dataset["rewards"] - reward.min()) / (reward.max() - reward.min())
    return dataset
    
def std_reward_normalizer(dataset):
    dataset["rewards"] = dataset["rewards"] / dataset["rewards"].std()
    return dataset

def get_mixed_d4rl_mujoco_datasets(agent, quality1, quality2, N, ratio, keep_traj=True, normalize_obs=False, normalize_reward=False):
    dataset_name1 = f"{agent}-{quality1}-v2"
    dataset_name2 = f"{agent}-{quality2}-v2"
    env, dataset1 = get_d4rl_dataset(dataset_name1)
    _, dataset2 = get_d4rl_dataset(dataset_name2)
    
    len1 = len(dataset1["observations"])
    len2 = len(dataset2["observations"])
    num1 = int(N * ratio)
    num2 = int(N * (1-ratio))
    
    if num1 and num2:
        if keep_traj:
            index1 = np.random.randint(len1)
            index1 = np.arange(index1, index1+num1) % len1
            index2 = np.random.randint(len2)
            index2 = np.arange(index2, index2+num2) % len2
        else:
            index1 = np.random.choice(len1, size=num1)
            index2 = np.random.choice(len2, size=num2)
        dataset = dict()
        for key in dataset1:
            dataset[key] = np.concatenate([dataset1[key][index1], dataset2[key][index2]], axis=0)
    else:
        if num1 == 0:
            dataset = dataset2
        else:
            dataset = dataset1
    
    if normalize_obs:
        dataset, info = _normalize_obs(dataset)
        from gym.wrappers.transform_observation import TransformObservation
        env = TransformObservation(env, lambda obs: (obs - info["obs_mean"])/info["obs_std"])
    if normalize_reward == "std":
        dataset = std_reward_normalizer(dataset)
    elif normalize_reward == "minmax":
        dataset = minmax_reward_normalizer(dataset)
    
    return env, dataset
        