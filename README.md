# RL Navigation Sandbox

A small reinforcement learning project built around a custom Gymnasium navigation environment and a PPO training script.

## Files

- `gym_env.py`: Gymnasium environment registration and `NavigationEnv` implementation.
- `train_ppo_nav.py`: CLI for training a PPO agent and replaying a saved checkpoint.
- `debug_env.py`: Quick smoke test for stepping through the environment.
- `simple_nav_env.py`: Standalone manual Pygame prototype for the navigation idea.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install gymnasium numpy pygame stable-baselines3
```

## Usage

Train a model:

```bash
python train_ppo_nav.py train --timesteps 5000 --run-name nav_run
```

Run inference with a saved checkpoint:

```bash
python train_ppo_nav.py infer --checkpoint models/nav_run_final.zip
```

Smoke test the environment:

```bash
python debug_env.py
```

Goal: use RL to create an agent that gets from point a to point b utilizing PPO and later adapt into ros2 and gazebo to control a robot
