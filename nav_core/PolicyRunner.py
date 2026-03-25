import numpy as np
from stable_baselines3 import PPO

class PolicyRunner:
    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(self, model_path):
        self.model = PPO.load(model_path, device="cpu")

    def build_observation(self, robot_x, robot_y, goal_x, goal_y):
        return np.array(
            [robot_x, robot_y, goal_x, goal_y],
            dtype=np.float32
        )

    def predict_action(self, robot_x, robot_y, goal_x, goal_y):
        obs = self.build_observation(robot_x, robot_y, goal_x, goal_y)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def predict_with_info(self, robot_x, robot_y, goal_x, goal_y):
        obs = self.build_observation(robot_x, robot_y, goal_x, goal_y)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        return {
            "observation": obs,
            "action": action,
            "action_name": self.action_name(action),
        }

    def action_name(self, action):
        return self.ACTION_NAMES.get(action, "unknown")