import gym

env = gym.make("CartPole-v0")
env.action_space.seed(42)

observation = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, info = env.step(env.action_space.sample())

    env.render()

    if terminated:
        observation = env.reset()

env.close()
