import gymnasium as gym
import gym_env  # registers NavigationEnv-v0

ENV_ID = "NavigationEnv-v0"


def main():
    env = gym.make(ENV_ID)
    obs, info = env.reset()
    print("reset:", obs, info)

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step} action={action} reward={reward:.2f} "
            f"terminated={terminated} truncated={truncated} obs={obs}"
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
