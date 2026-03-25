import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import register, spaces


ENV_ID = "NavigationEnv-v0"

register(
    id=ENV_ID,
    entry_point=f"{__name__}:NavigationEnv",
    max_episode_steps=300,
)


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()

        self.env_id = "NavigationEnv-v0"
        self.window_size = 600
        self.robot_size = 20
        self.goal_size = 20
        self.robot_speed = 5
        self.padding = 20
        self.success_distance = 10
        self.min_goal_distance = 100
        self.max_episode_steps = 300
        self.step_penalty = 1.0
        self.closer_reward = 1.0
        self.further_penalty = 1.0
        self.goal_reward = 1000.0
        self.reward_truncation_threshold = -100.0
        self.truncation_penalty = 50.0

        self.width = self.window_size
        self.height = self.window_size

        self.robot_x = 100
        self.robot_y = 100

        self.goal_x = 400
        self.goal_y = 400

        self.window = None
        self.clock = None

        self.steps = 0
        self.start_time = 0.0
        self.episode_reward = 0.0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.width, self.height, self.width, self.height], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self.robot_x = self.np_random.integers(
            self.padding,
            self.width - self.robot_size - self.padding + 1,
        )
        self.robot_y = self.np_random.integers(
            self.padding,
            self.height - self.robot_size - self.padding + 1,
        )

        while True:
            self.goal_x = self.np_random.integers(
                self.padding,
                self.width - self.goal_size - self.padding + 1,
            )
            self.goal_y = self.np_random.integers(
                self.padding,
                self.height - self.goal_size - self.padding + 1,
            )

            robot_cx = self.robot_x + self.robot_size / 2
            robot_cy = self.robot_y + self.robot_size / 2
            goal_cx = self.goal_x + self.goal_size / 2
            goal_cy = self.goal_y + self.goal_size / 2
            distance_to_goal = math.sqrt((goal_cx - robot_cx) ** 2 + (goal_cy - robot_cy) ** 2)
            if distance_to_goal > self.min_goal_distance:
                break

        self.steps = 0
        self.start_time = time.time()
        self.episode_reward = 0.0

        observation = np.array(
            [self.robot_x, self.robot_y, self.goal_x, self.goal_y],
            dtype=np.float32,
        )
        return observation, {}

    def step(self, action):
        self.steps += 1

        action = int(action)
        robot_cx = self.robot_x + self.robot_size / 2
        robot_cy = self.robot_y + self.robot_size / 2
        goal_cx = self.goal_x + self.goal_size / 2
        goal_cy = self.goal_y + self.goal_size / 2
        old_distance = math.sqrt((goal_cx - robot_cx) ** 2 + (goal_cy - robot_cy) ** 2)

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

        robot_cx = self.robot_x + self.robot_size / 2
        robot_cy = self.robot_y + self.robot_size / 2
        goal_cx = self.goal_x + self.goal_size / 2
        goal_cy = self.goal_y + self.goal_size / 2
        new_distance = math.sqrt((goal_cx - robot_cx) ** 2 + (goal_cy - robot_cy) ** 2)
        reward, terminated, truncated, termination_reason = self.reward(old_distance, new_distance)

        observation = np.array(
            [self.robot_x, self.robot_y, self.goal_x, self.goal_y],
            dtype=np.float32,
        )
        return observation, reward, terminated, truncated, {
            "distance_to_goal": round(new_distance, 3),
            "termination_reason": termination_reason,
        }

    def reward(self, old_distance, new_distance):
        termination_reason = "running"
        reward = -self.step_penalty
        if new_distance < old_distance:
            reward += self.closer_reward
        else:
            reward -=self.further_penalty

        at_goal = (
            self.robot_x <= self.goal_x + self.goal_size
            and self.robot_x + self.robot_size >= self.goal_x
            and self.robot_y <= self.goal_y + self.goal_size
            and self.robot_y + self.robot_size >= self.goal_y
        )

        terminated = False
        if at_goal:
            reward += self.goal_reward
            terminated = True
            termination_reason = "goal"

        self.episode_reward += reward

        truncated = False
        if self.episode_reward <= self.reward_truncation_threshold:
            truncated = True
            termination_reason = "reward_floor"
        elif self.steps >= self.max_episode_steps:
            truncated = True
            termination_reason = "time_limit"

        if truncated and not terminated:
            reward -= self.truncation_penalty
            self.episode_reward -= self.truncation_penalty

        return reward, terminated, truncated, termination_reason

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
