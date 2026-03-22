import sys

print("=== Python Environment Check ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# pygame
try:
    import pygame
    print(f"pygame: OK ({pygame.__version__})")
except Exception as e:
    print(f"pygame: FAILED -> {e}")

# numpy
try:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f"numpy: OK ({np.__version__}) | test array sum = {arr.sum()}")
except Exception as e:
    print(f"numpy: FAILED -> {e}")

# matplotlib
try:
    import matplotlib
    print(f"matplotlib: OK ({matplotlib.__version__})")
except Exception as e:
    print(f"matplotlib: FAILED -> {e}")

# gymnasium
try:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print(f"gymnasium: OK ({gym.__version__}) | CartPole reset successful")
    env.close()
except Exception as e:
    print(f"gymnasium: FAILED -> {e}")

# stable-baselines3
try:
    import stable_baselines3
    from stable_baselines3 import PPO
    print(f"stable_baselines3: OK ({stable_baselines3.__version__})")
except Exception as e:
    print(f"stable_baselines3: FAILED -> {e}")

# torch
try:
    import torch
    print(f"torch: OK ({torch.__version__})")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"torch tensor test sum = {x.sum().item()}")
except Exception as e:
    print(f"torch: FAILED -> {e}")

# tensorboard
try:
    import tensorboard
    print(f"tensorboard: OK ({tensorboard.__version__})")
except Exception as e:
    print(f"tensorboard: FAILED -> {e}")

print()
print("=== Done ===")