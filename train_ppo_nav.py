from __future__ import annotations

import argparse
from pathlib import Path
import time

import gymnasium as gym
import gym_env
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

CHECKPOINT_DIR = Path("checkpoints")
MODEL_DIR = Path("models")
TENSORBOARD_DIR = Path("runs")


def make_env():
    return Monitor(
        gym.make(gym_env.ENV_ID),
        info_keywords=("distance_to_goal", "termination_reason"),
    )


def train(total_timesteps=5000, run_name="debug_run", save_freq=1000, resume_checkpoint=None):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    TENSORBOARD_DIR.mkdir(exist_ok=True)

    env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix=run_name,
    )

    if resume_checkpoint:
        model = PPO.load(resume_checkpoint, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR),
            n_steps=512,
            batch_size=128,
            learning_rate=1e-4,
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
        reset_num_timesteps=not bool(resume_checkpoint),
        progress_bar=True,
    )

    final_model_path = MODEL_DIR / f"{run_name}_final"
    model.save(str(final_model_path))
    env.close()

    return f"{final_model_path}.zip"


def run_inference(checkpoint_path, delay=0.03):
    model = PPO.load(checkpoint_path, device="cpu")
    env = gym.make(gym_env.ENV_ID)

    episode_idx = 0
    while True:
        obs, info = env.reset()
        done = False
        episode_idx += 1
        step_idx = 0
        episode_reward = 0.0

        while not done:
            if pygame.get_init() and pygame.display.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1
            episode_reward += reward
            print(
                f"[infer] episode={episode_idx} step={step_idx} "
                f"reward={reward:.3f} episode_reward={episode_reward:.3f}"
            )
            env.render()
            time.sleep(delay)

        print(
            f"[infer] episode={episode_idx} finished "
            f"total_reward={episode_reward:.3f} "
            f"termination={info.get('termination_reason', 'unknown')}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train or run the navigation PPO agent.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train_parser.add_argument("--timesteps", type=int, default=5000)
    train_parser.add_argument("--run-name", type=str, default="debug_run")
    train_parser.add_argument("--save-freq", type=int, default=1000)
    train_parser.add_argument("--resume-checkpoint", type=str, default=None)

    infer_parser = subparsers.add_parser("infer", help="Run a trained agent.")
    infer_parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "train":
        final_model = train(
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            save_freq=args.save_freq,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(f"Saved model to {final_model}")
        run_inference(final_model)

    else:
        run_inference(args.checkpoint)


if __name__ == "__main__":
    main()
