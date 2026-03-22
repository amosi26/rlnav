import math
import time

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import register, spaces

register(
    id="NavigationEnv-v0",
    entry_point="gym_env:NavigationEnv",
    max_episode_steps=400,
)


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()

        self.width = 600
        self.height = 600

        self.robot_x = 100
        self.robot_y = 100
        self.robot_speed = 3

        self.goal_x = 400
        self.goal_y = 400

        self.robot_size = 20
        self.goal_size = 20

        self.window = None
        self.clock = None

        self.steps = 0
        self.start_time = 0.0

        self.prev_action = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.width, self.height, self.width, self.height], dtype=np.float32),
            dtype=np.float32,
        )

    def _distance_to_goal(self):
        robot_cx = self.robot_x + self.robot_size / 2
        robot_cy = self.robot_y + self.robot_size / 2
        goal_cx = self.goal_x + self.goal_size / 2
        goal_cy = self.goal_y + self.goal_size / 2

        return math.sqrt((goal_cx - robot_cx) ** 2 + (goal_cy - robot_cy) ** 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_x = self.np_random.integers(20, self.width - self.robot_size - 20 + 1)
        self.robot_y = self.np_random.integers(20, self.height - self.robot_size - 20 + 1)

        while True:
            self.goal_x = self.np_random.integers(20, self.width - self.goal_size - 20 + 1)
            self.goal_y = self.np_random.integers(20, self.height - self.goal_size - 20 + 1)

            if self._distance_to_goal() > 100:
                break

        self.steps = 0
        self.start_time = time.time()
        self.prev_action = None

        observation = np.array(
            [self.robot_x, self.robot_y, self.goal_x, self.goal_y],
            dtype=np.float32,
        )

        info = {}
        return observation, info

    def step(self, action):
        self.steps += 1

        old_x = self.robot_x
        old_y = self.robot_y
        old_distance = self._distance_to_goal()

        if action == 0:      # up
            self.robot_y -= self.robot_speed
        elif action == 1:    # down
            self.robot_y += self.robot_speed
        elif action == 2:    # left
            self.robot_x -= self.robot_speed
        elif action == 3:    # right
            self.robot_x += self.robot_speed

        self.robot_x = max(0, min(self.robot_x, self.width - self.robot_size))
        self.robot_y = max(0, min(self.robot_y, self.height - self.robot_size))

        new_distance = self._distance_to_goal()

        # Strong reward for reducing distance
        progress_reward = (old_distance - new_distance) * 3.0

        # Extra bonus when close so it keeps pushing inward
        proximity_bonus = max(0.0, (60.0 - new_distance) * 0.15)

        if new_distance < 10:
            reward = 1000
            terminated = True
        else:
            reward = -1 + progress_reward + proximity_bonus
            terminated = False

        # Penalize no movement at all
        if self.robot_x == old_x and self.robot_y == old_y:
            reward -= 3

        # Penalize immediate opposite-action oscillation
        opposite_action = (
            (self.prev_action == 0 and action == 1) or
            (self.prev_action == 1 and action == 0) or
            (self.prev_action == 2 and action == 3) or
            (self.prev_action == 3 and action == 2)
        )
        if opposite_action:
            reward -= 2
            if new_distance < 40:
                reward -= 4

        # Penalize weak progress:
        # if distance does not decrease by at least 5% this step, punish it
        if old_distance > 0:
            percent_improvement = (old_distance - new_distance) / old_distance
            if percent_improvement < 0.05:
                reward -= 4

        truncated = self.steps >= 400

        observation = np.array(
            [self.robot_x, self.robot_y, self.goal_x, self.goal_y],
            dtype=np.float32,
        )

        info = {}

        self.prev_action = action

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Navigation Environment")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        pygame.draw.rect(
            self.window,
            (0, 0, 255),
            (self.robot_x, self.robot_y, self.robot_size, self.robot_size),
        )

        pygame.draw.rect(
            self.window,
            (0, 200, 0),
            (self.goal_x, self.goal_y, self.goal_size, self.goal_size),
        )

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None