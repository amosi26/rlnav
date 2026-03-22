import gymnasium as gym
import gym_env
import time

env = gym.make("NavigationEnv-v0")

obs, info = env.reset()
print("Start obs:", obs)

for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(f"Step {i+1}, Action: {action}, Reward: {reward}")

    if terminated or truncated:
        print("Episode ended")
        break

    time.sleep(0.05)

env.close()
