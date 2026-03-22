from __future__ import annotations

import argparse
from pathlib import Path
import time

import gymnasium as gym
import gym_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

CHECKPOINT_DIR = Path("checkpoints")
MODEL_DIR = Path("models")
TENSORBOARD_DIR = Path("runs")


def make_env():
    return Monitor(
        gym.make(gym_env.ENV_ID),
        info_keywords=("distance_to_goal", "termination_reason", "no_progress_steps", "flip_streak"),
    )


def train(total_timesteps=5000, run_name="debug_run", save_freq=1000):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    TENSORBOARD_DIR.mkdir(exist_ok=True)

    env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix=run_name,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        n_steps=512,
        batch_size=256,
        learning_rate=1.5e-4,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.003,
        device="cpu",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        progress_bar=True,
    )

    final_model_path = MODEL_DIR / f"{run_name}_final"
    model.save(str(final_model_path))
    env.close()

    return f"{final_model_path}.zip"


def run_inference(checkpoint_path, episodes=3, delay=0.03):
    model = PPO.load(checkpoint_path)
    env = gym.make(gym_env.ENV_ID)

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
    parser = argparse.ArgumentParser(description="Train or run the navigation PPO agent.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train_parser.add_argument("--timesteps", type=int, default=5000)
    train_parser.add_argument("--run-name", type=str, default="debug_run")
    train_parser.add_argument("--save-freq", type=int, default=1000)

    infer_parser = subparsers.add_parser("infer", help="Run a trained agent.")
    infer_parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "train":
        final_model = train(
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            save_freq=args.save_freq,
        )
        print(f"Saved model to {final_model}")
        run_inference(final_model)

    else:
        run_inference(args.checkpoint)


if __name__ == "__main__":
    main()
