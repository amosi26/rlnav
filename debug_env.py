import gymnasium as gym
import gym_env  # registers NavigationEnv-v0

env = gym.make("NavigationEnv-v0")

print("made env")

obs, info = env.reset()
print("reset ok:", obs, info)

for i in range(5):
    action = env.action_space.sample()
    print(f"step {i}, action={action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print("returned:", obs, reward, terminated, truncated, info)

env.close()
print("done")