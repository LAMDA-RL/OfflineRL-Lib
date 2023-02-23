import torch
import gym
import torch.nn as nn
import numpy as np

@torch.no_grad()
def eval_offline_policy(
    env: gym.Env, actor: nn.Module, n_episodes: int, seed: int, score_func=None
):
    if score_func is None:
        score_func = env.get_normalized_score
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
        episode_rewards.append(score_func(episode_reward)*100)
        episode_lengths.append(episode_length)

    actor.train()
    episode_rewards = np.asarray(episode_rewards)
    episode_lengths = np.asarray(episode_lengths)
    return {
        "normalized_score_mean": episode_rewards.mean(), 
        "normalized_score_std": episode_rewards.std(), 
        "length_mean": episode_lengths.mean(), 
        "length_std": episode_lengths.std()
    }
    
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
