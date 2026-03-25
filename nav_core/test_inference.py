from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = Path(__file__).resolve().parent / "models" / "finalmodel.zip"

action_names = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

model = PPO.load(str(MODEL_PATH), device="cpu")
print("loaded")

obs = np.array([100.0, 100.0, 400.0, 400.0], dtype=np.float32)

action, _ = model.predict(obs, deterministic=True)
action = int(action)

print("obs:", obs)
print("predicted action:", action)
print("meaning:", action_names[action])
