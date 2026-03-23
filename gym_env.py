import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import register, spaces


register(
    id="NavigationEnv-v0",
    entry_point="gym_env:NavigationEnv",
    max_episode_steps=30,
)


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()

        self.env_id = "NavigationEnv-v0"
        self.window_size = 600
        self.robot_size = 20
        self.goal_size = 20
        self.robot_speed = 3
        self.padding = 20
        self.success_distance = 10
        self.min_goal_distance = 100
        self.max_episode_steps = 30
        self.max_distance = math.sqrt(2 * (self.window_size ** 2))
        self.step_penalty = 0.05
        self.progress_scale = 3.0
        self.success_reward = 100.0
        self.wall_penalty = 1.5
        self.turn_penalty = 0.4
        self.oscillation_penalty = 0.75
        self.no_progress_penalty = 0.25
        self.progress_epsilon = 0.5
        self.no_progress_limit = 18
        self.stuck_limit = 4
        self.flip_limit = 6

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

        self.prev_action = None
        self.no_progress_steps = 0
        self.stuck_steps = 0
        self.flip_streak = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([self.width, self.height, self.width, self.height, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def _goal_features(self) -> tuple[float, float]:
        robot_cx = self.robot_x + self.robot_size / 2
        robot_cy = self.robot_y + self.robot_size / 2
        goal_cx = self.goal_x + self.goal_size / 2
        goal_cy = self.goal_y + self.goal_size / 2

        dx = goal_cx - robot_cx
        dy = goal_cy - robot_cy
        distance = math.sqrt((dx ** 2) + (dy ** 2))
        heading = math.atan2(dy, dx) / math.pi
        return distance / self.max_distance, heading

    def _get_observation(self) -> np.ndarray:
        distance_norm, heading = self._goal_features()
        return np.array(
            [self.robot_x, self.robot_y, self.goal_x, self.goal_y, distance_norm, heading],
            dtype=np.float32,
        )

    def _distance_to_goal(self) -> float:
        robot_cx = self.robot_x + self.robot_size / 2
        robot_cy = self.robot_y + self.robot_size / 2
        goal_cx = self.goal_x + self.goal_size / 2
        goal_cy = self.goal_y + self.goal_size / 2

        return math.sqrt((goal_cx - robot_cx) ** 2 + (goal_cy - robot_cy) ** 2)

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

            if self._distance_to_goal() > self.min_goal_distance:
                break

        self.steps = 0
        self.start_time = time.time()
        self.prev_action = None
        self.no_progress_steps = 0
        self.stuck_steps = 0
        self.flip_streak = 0

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        requested_action = int(action)
        old_x = self.robot_x
        old_y = self.robot_y
        old_distance = self._distance_to_goal()
        termination_reason = "running"

        opposite_action = (
            (self.prev_action == 0 and requested_action == 1) or
            (self.prev_action == 1 and requested_action == 0) or
            (self.prev_action == 2 and requested_action == 3) or
            (self.prev_action == 3 and requested_action == 2)
        )
        if opposite_action and self.flip_streak >= 2:
            action = self.prev_action
        else:
            action = requested_action

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

        moved = self.robot_x != old_x or self.robot_y != old_y
        wall_hit = not moved
        new_distance = self._distance_to_goal()
        progress_delta = old_distance - new_distance
        reward = (self.progress_scale * progress_delta) - self.step_penalty

        if new_distance < self.success_distance:
            reward += self.success_reward
            terminated = True
            termination_reason = "goal"
        else:
            terminated = False

        if wall_hit:
            reward -= self.wall_penalty
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if opposite_action:
            self.flip_streak += 1
            reward -= self.oscillation_penalty
        else:
            self.flip_streak = 0

        changed_axis = self.prev_action is not None and (action // 2) != (self.prev_action // 2)
        if changed_axis and progress_delta <= self.progress_epsilon:
            reward -= self.turn_penalty

        if progress_delta <= self.progress_epsilon:
            self.no_progress_steps += 1
            reward -= self.no_progress_penalty
        else:
            self.no_progress_steps = 0

        if not terminated and self.stuck_steps >= self.stuck_limit:
            terminated = True
            reward -= self.wall_penalty
            termination_reason = "stuck"

        if not terminated and self.flip_streak >= self.flip_limit:
            terminated = True
            reward -= self.oscillation_penalty
            termination_reason = "oscillation"

        if not terminated and self.no_progress_steps >= self.no_progress_limit:
            terminated = True
            reward -= self.no_progress_penalty
            termination_reason = "no_progress"

        truncated = self.steps >= self.max_episode_steps
        if truncated:
            termination_reason = "time_limit"

        self.prev_action = action

        return self._get_observation(), reward, terminated, truncated, {
            "distance_to_goal": round(new_distance, 3),
            "termination_reason": termination_reason,
            "no_progress_steps": self.no_progress_steps,
            "flip_streak": self.flip_streak,
        }

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
