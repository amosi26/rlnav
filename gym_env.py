import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import register, spaces

ENV_ID = "NavigationEnv-v0"
WINDOW_SIZE = 600
ROBOT_SIZE = 20
GOAL_SIZE = 20
ROBOT_SPEED = 3
PADDING = 20
SUCCESS_DISTANCE = 10
MIN_GOAL_DISTANCE = 100
MAX_EPISODE_STEPS = 150
MAX_DISTANCE = math.sqrt(2 * (WINDOW_SIZE ** 2))
STEP_PENALTY = 0.05
PROGRESS_SCALE = 3.0
SUCCESS_REWARD = 100.0
WALL_PENALTY = 1.5
TURN_PENALTY = 0.4
OSCILLATION_PENALTY = 0.75
NO_PROGRESS_PENALTY = 0.25
PROGRESS_EPSILON = 0.5
NO_PROGRESS_LIMIT = 18
STUCK_LIMIT = 4
FLIP_LIMIT = 6

register(
    id=ENV_ID,
    entry_point="gym_env:NavigationEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()

        self.width = WINDOW_SIZE
        self.height = WINDOW_SIZE

        self.robot_x = 100
        self.robot_y = 100
        self.robot_speed = ROBOT_SPEED

        self.goal_x = 400
        self.goal_y = 400

        self.robot_size = ROBOT_SIZE
        self.goal_size = GOAL_SIZE

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
        return distance / MAX_DISTANCE, heading

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
            PADDING,
            self.width - self.robot_size - PADDING + 1,
        )
        self.robot_y = self.np_random.integers(
            PADDING,
            self.height - self.robot_size - PADDING + 1,
        )

        while True:
            self.goal_x = self.np_random.integers(
                PADDING,
                self.width - self.goal_size - PADDING + 1,
            )
            self.goal_y = self.np_random.integers(
                PADDING,
                self.height - self.goal_size - PADDING + 1,
            )

            if self._distance_to_goal() > MIN_GOAL_DISTANCE:
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
        reward = (PROGRESS_SCALE * progress_delta) - STEP_PENALTY

        if new_distance < SUCCESS_DISTANCE:
            reward += SUCCESS_REWARD
            terminated = True
            termination_reason = "goal"
        else:
            terminated = False

        if wall_hit:
            reward -= WALL_PENALTY
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if opposite_action:
            self.flip_streak += 1
            reward -= OSCILLATION_PENALTY
        else:
            self.flip_streak = 0

        changed_axis = self.prev_action is not None and (action // 2) != (self.prev_action // 2)
        if changed_axis and progress_delta <= PROGRESS_EPSILON:
            reward -= TURN_PENALTY

        if progress_delta <= PROGRESS_EPSILON:
            self.no_progress_steps += 1
            reward -= NO_PROGRESS_PENALTY
        else:
            self.no_progress_steps = 0

        if not terminated and self.stuck_steps >= STUCK_LIMIT:
            terminated = True
            reward -= WALL_PENALTY
            termination_reason = "stuck"

        if not terminated and self.flip_streak >= FLIP_LIMIT:
            terminated = True
            reward -= OSCILLATION_PENALTY
            termination_reason = "oscillation"

        if not terminated and self.no_progress_steps >= NO_PROGRESS_LIMIT:
            terminated = True
            reward -= NO_PROGRESS_PENALTY
            termination_reason = "no_progress"

        truncated = self.steps >= MAX_EPISODE_STEPS
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
