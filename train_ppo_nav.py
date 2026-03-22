from __future__ import annotations

import argparse
import time

import gymnasium as gym
import gym_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def make_env():
    env = gym.make("NavigationEnv-v0")
    env = Monitor(env)
    return env


def train(total_timesteps=5000, run_name="debug_run", save_freq=1000):
    print("before make_env")
    env = make_env()
    print("after make_env")

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="checkpoints",
        name_prefix=run_name,
    )

    print("before model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="runs",
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
    )
    print("after model")

    print("before learn")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        progress_bar=True,
    )
    print("after learn")

    model.save(f"models/{run_name}_final")
    env.close()

    return f"models/{run_name}_final.zip"


def run_inference(checkpoint_path, episodes=3, delay=0.03):
    model = PPO.load(checkpoint_path)
    env = gym.make("NavigationEnv-v0")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
            time.sleep(delay)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--timesteps", type=int, default=5000)
    train_parser.add_argument("--run-name", type=str, default="debug_run")
    train_parser.add_argument("--save-freq", type=int, default=1000)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "train":
        final_model = train(
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            save_freq=args.save_freq,
        )
        print("running trained agent")
        run_inference(final_model)

    elif args.mode == "infer":
        run_inference(args.checkpoint)


if __name__ == "__main__":
    main()
