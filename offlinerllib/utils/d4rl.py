import numpy as np

import gym
import d4rl

from offlinerllib.buffer.dataset import D4RLDataset

def _fix_terminal(dataset):
    terminal = dataset["terminals"]
    count = 0
    for i in range(len(terminal) - 1):
        if terminal[i]:
            continue
        elif np.linalg.norm(dataset["observations"][i+1] - dataset["next_observations"][i]) > 1e-6:
            terminal[i] = True
            count += 1
    terminal[-1] = True
    dataset["terminals"] = terminal
    print(f"fixed {count} terminals")
    return dataset, {}
    
def _antmaze_normalize_reward(dataset):
    dataset["rewards"] -= 1
    return dataset, {}
    
def _normalize_reward(dataset):
    def split_into_trajs(obs, action, next_obs, reward, terminal):
        trajs = [[]]
        total_num = len(obs)
        for i in range(total_num):
            trajs[-1].append((obs[i], action[i], reward[i], terminal[i], next_obs[i]))
            if terminal[i] and i+1 < total_num:
                trajs.append([])
        return trajs
    def compute_return(traj):
        episode_return = 0
        for _, _, rew, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs = split_into_trajs(dataset["observations"], dataset["actions"], dataset["next_observations"], dataset["rewards"], dataset["terminals"])
    trajs.sort(key=compute_return)
    dataset["rewards"] /= compute_return(trajs[-1]) - compute_return(trajs[0])
    dataset["rewards"] *= 1000
    return dataset, {}
    
def _normalize_obs(dataset):
    all_obs = np.concatenate([dataset["observations"], dataset["next_observations"]], axis=0)
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0)+1e-6
    dataset["observations"] = (dataset["observations"] - obs_mean) / obs_std
    dataset["next_observations"] = (dataset["next_observations"] - obs_mean) / obs_std
    return dataset, {
        "obs_mean": obs_mean, 
        "obs_std": obs_std
    }
        
def get_d4rl_dataset(task, fix_terminal=False, normalize_reward=False, normalize_obs=False):
    env = gym.make(task)
    dataset = d4rl.qlearning_dataset(env)
    if fix_terminal:
        dataset, _ = _fix_terminal(dataset)
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
        