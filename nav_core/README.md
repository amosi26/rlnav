# RL Navigation Sandbox

A small reinforcement learning project built around a custom Gymnasium navigation environment, PPO training, and standalone policy validation scripts.

## Files

- `gym_env.py`: Gymnasium environment registration and `NavigationEnv` implementation.
- `train_ppo_nav.py`: CLI for training a PPO agent and replaying a saved checkpoint.
- `PolicyRunner.py`: Lightweight PPO inference wrapper.
- `test_inference.py`: Single-observation model load/inference smoke test.
- `run_policy_tests.py`: Multi-scenario directional behavior checks.
- `simple_nav_env.py`: Standalone manual Pygame prototype for the navigation idea.

## Setup

```bash
cd ..
python -m venv .venv
source .venv/bin/activate
pip install gymnasium numpy pygame stable-baselines3
```

## Usage

Run standalone inference smoke test (from repo root):

```bash
python nav_core/test_inference.py
```

Run directional policy tests (from repo root):

```bash
python nav_core/run_policy_tests.py
```

Train or replay from within `nav_core/`:

```bash
cd nav_core
python train_ppo_nav.py train --timesteps 5000 --run-name nav_run
python train_ppo_nav.py infer --checkpoint models/nav_run_final.zip
```

Model location:

- `models/finalmodel.zip`
