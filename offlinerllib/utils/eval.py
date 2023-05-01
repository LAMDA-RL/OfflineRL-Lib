from typing import Callable, Dict, List

import gym
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def eval_offline_policy(
    env: gym.Env, actor: nn.Module, n_episodes: int, seed: int, score_func=None
) -> Dict[str, float]:
    if score_func is None:
        score_func = env.get_normalized_score
    env.seed(seed)
    actor.eval()
    # actor.to(device)
    episode_lengths = []
    episode_returns = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_return = 0.0
        episode_length = 0.0
        while not done:
            action = actor.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_return += reward
            episode_length += 1
        episode_returns.append(score_func(episode_return)*100)
        episode_lengths.append(episode_length)

    actor.train()
    episode_returns = np.asarray(episode_returns)
    episode_lengths = np.asarray(episode_lengths)
    return {
        "normalized_score_mean": episode_returns.mean(), 
        "normalized_score_std": episode_returns.std(), 
        "length_mean": episode_lengths.mean(), 
        "length_std": episode_lengths.std()
    }
    
@torch.no_grad()
def eval_decision_transformer(
    env: gym.Env, actor: nn.Module, target_returns: List[float], return_scale: float, n_episode: int, seed: int, score_func=None
) -> Dict[str, float]:
    
    def eval_one_return(target_return, score_func=None):
        if score_func is None:
            score_func = env.get_normalized_score
        env.seed(seed)
        actor.eval()
        episode_lengths = []
        episode_returns = []
        timesteps = np.arange(actor.episode_len, dtype=int)[None, :]
        for _ in range(n_episode):
            states = np.zeros([1, actor.episode_len+1, actor.state_dim], dtype=np.float32)
            actions = np.zeros([1, actor.episode_len+1, actor.action_dim], dtype=np.float32)
            returns_to_go = np.zeros([1, actor.episode_len+1, 1], dtype=np.float32)
            state, done = env.reset(), False
            
            states[:, 0] = state
            returns_to_go[:, 0] = target_return*return_scale
            
            episode_return = episode_length = 0
            for step in range(actor.episode_len):
                action = actor.select_action(
                    states[:, :step+1][:, -actor.seq_len: ], 
                    actions[:, :step+1][:, -actor.seq_len: ], 
                    returns_to_go[:, :step+1][:, -actor.seq_len: ], 
                    timesteps[:, :step+1][:, -actor.seq_len: ]
                )
                next_state, reward, done, info = env.step(action)
                actions[:, step] = action
                states[:, step+1] = next_state
                returns_to_go[:, step+1] = returns_to_go[:, step] - reward*return_scale
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    episode_returns.append(score_func(episode_return)*100)
                    episode_lengths.append(episode_length)
                    break
                
        actor.train()
        episode_returns = np.asarray(episode_returns)
        episode_lengths = np.asarray(episode_lengths)
        return {
            "normalized_score_mean_target{:.1f}".format(target_return): episode_returns.mean(), 
            "normalized_score_std_target{:.1f}".format(target_return): episode_returns.std(), 
            "length_mean_target{:.1f}".format(target_return): episode_lengths.mean(), 
            "length_std_target{:.1f}".format(target_return): episode_lengths.std()
        }
    
    ret = {}
    for target in target_returns:
        ret.update(eval_one_return(target, score_func))
    return ret
    
            
@torch.no_grad()
def eval_online_policy(
    env: gym.Env, actor: nn.Module, n_episodes: int, seed: int
):
    env.seed(seed)
    actor.eval()
    # actor.to(device)
    episode_lengths = []
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0.0
        while not done:
            action = actor.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    actor.train()
    episode_rewards = np.asarray(episode_rewards)
    episode_lengths = np.asarray(episode_lengths)
    return {
        "episode_return_mean": episode_rewards.mean(), 
        "episode_return_std": episode_rewards.std(), 
        "length_mean": episode_lengths.mean(), 
        "length_std": episode_lengths.std()
    }
