import math
from pathlib import Path
import sys

# Find the repository root by looking for nav_core/PolicyRunner.py
def _find_repo_root() -> Path:
    marker = Path("nav_core") / "PolicyRunner.py"
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / marker).exists():
            return parent
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / marker).exists():
            return parent
    return current.parents[3]

REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add virtual environment site-packages to sys.path for dependencies
def _add_venv_site_packages():
    venv_lib = REPO_ROOT / ".venv" / "lib"
    if not venv_lib.exists():
        return
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_str = str(site_packages)
        if site_str not in sys.path:
            sys.path.insert(0, site_str)

_add_venv_site_packages()

# ROS2 and navigation imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry
from nav_core.PolicyRunner import PolicyRunner
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
import random
from ros2_policy_runner.srv import SetGoal


# Configuration constants
MODEL_PATH = REPO_ROOT / "nav_core" / "models" / "finalmodel.zip"
GOAL_THRESHOLD = 0.5
LINEAR_SPEED = 0.25
ANGULAR_SPEED = 0.7
TIMER_PERIOD = 0.1
GRID_MIN = 0.0
GRID_MAX = 8.0

# Convert discrete action to ROS Twist message for planar base motion
def action_to_twist(action):
    twist = Twist()
    # Mapping matches gym_env.py: 0=up, 1=down, 2=left, 3=right
    # Pygame "down" increases y, so we map down -> +y
    if action == 0:
        twist.linear.y = -LINEAR_SPEED
    elif action == 1:
        twist.linear.y = LINEAR_SPEED
    elif action == 2:
        twist.linear.x = -LINEAR_SPEED
    elif action == 3:
        twist.linear.x = LINEAR_SPEED
    return twist

# Main ROS2 node class that runs the navigation policy

class PolicyRunnerNode(Node):
    def __init__(self):
        super().__init__("policy_runner_node")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Service to set a new goal
        self.goal_srv = self.create_service(
            SetGoal,
            "set_goal",
            self.set_goal_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        # Robot state variables
        self.robot_x = None
        self.robot_y = None
        self.robot_initialized = False

        # Goal state
        self.goal_x = None
        self.goal_y = None
        self.declare_parameter("auto_randomize_goal", True)
        self.auto_randomize_goal = bool(self.get_parameter("auto_randomize_goal").value)
        self.declare_parameter("obs_scale", 75.0)
        self.obs_scale = float(self.get_parameter("obs_scale").value)

        # Load the trained PPO policy model
        model_path = MODEL_PATH.resolve()
        self.get_logger().info(f"Loading policy from: {model_path}")
        self.runner = PolicyRunner(str(model_path))
        self.get_logger().info(f"Observation scale: {self.obs_scale}")

        # Timer to periodically run policy inference
        self.create_timer(TIMER_PERIOD, self.timer_callback)

        # Randomize robot start position (simulate by waiting for first odom)
        self.declare_parameter("randomize_robot_start", True)
        self.randomize_robot_start = bool(self.get_parameter("randomize_robot_start").value)
        self.spawned_goal = False

    def odom_callback(self, msg):
        if self.randomize_robot_start and not self.robot_initialized:
            # Only randomize at startup
            self.robot_x = random.uniform(GRID_MIN, GRID_MAX)
            self.robot_y = random.uniform(GRID_MIN, GRID_MAX)
            self.robot_initialized = True
            self.get_logger().info(f"Robot spawned at random: ({self.robot_x:.2f}, {self.robot_y:.2f})")
        else:
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y

    def set_goal_callback(self, request, response):
        self.goal_x = request.x
        self.goal_y = request.y
        self.spawned_goal = True
        response.success = True
        response.message = f"Goal set to ({self.goal_x:.2f}, {self.goal_y:.2f})"
        self.get_logger().info(response.message)
        return response

    def _set_random_goal(self, reason: str):
        self.goal_x = random.uniform(GRID_MIN, GRID_MAX)
        self.goal_y = random.uniform(GRID_MIN, GRID_MAX)
        self.spawned_goal = True
        self.get_logger().info(
            f"{reason}: new goal=({self.goal_x:.2f}, {self.goal_y:.2f})"
        )

    def timer_callback(self):
        if self.robot_x is None or self.robot_y is None:
            return

        if self.goal_x is None or self.goal_y is None:
            return

        if self.auto_randomize_goal and not self.spawned_goal:
            self._set_random_goal("Auto-randomize")

        dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

        if dist < GOAL_THRESHOLD:
            cmd = Twist()
            reason = "at goal, stop"
            if self.auto_randomize_goal:
                self._set_random_goal("Reached goal")
            else:
                # Wait for new goal to be set via service
                self.spawned_goal = False
        else:
            action = self.runner.predict_action(
                self.robot_x * self.obs_scale,
                self.robot_y * self.obs_scale,
                self.goal_x * self.obs_scale,
                self.goal_y * self.obs_scale,
            )
            action_name = self.runner.action_name(action)
            cmd = action_to_twist(action)
            reason = f"action {action} ({action_name})"

        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f"obs=[{self.robot_x:.2f},{self.robot_y:.2f},{self.goal_x:.2f},{self.goal_y:.2f}] "
            f"dist={dist:.2f} -> {reason}"
        )

# Main function to initialize and run the ROS2 node
def main(args=None):
    rclpy.init(args=args)
    node = PolicyRunnerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
