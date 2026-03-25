import math
from pathlib import Path
import sys

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


def _add_venv_site_packages():
    venv_lib = REPO_ROOT / ".venv" / "lib"
    if not venv_lib.exists():
        return

    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_str = str(site_packages)
        if site_str not in sys.path:
            sys.path.insert(0, site_str)


_add_venv_site_packages()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from nav_core.PolicyRunner import PolicyRunner

# Constants to make behavior and tuning easy to read
MODEL_PATH = REPO_ROOT / "nav_core" / "models" / "finalmodel.zip"
GOAL_THRESHOLD = 0.5
LINEAR_SPEED = 0.25
ANGULAR_SPEED = 0.7
TIMER_PERIOD = 1.0  # seconds


def action_to_twist(action):
    """Map discrete actions to simple Twist commands."""
    twist = Twist()

    if action == 0:  # up => forward
        twist.linear.x = LINEAR_SPEED
    elif action == 1:  # down => backward
        twist.linear.x = -LINEAR_SPEED
    elif action == 2:  # left => yaw left
        twist.angular.z = ANGULAR_SPEED
    elif action == 3:  # right => yaw right
        twist.angular.z = -ANGULAR_SPEED
    else:
        twist = Twist()

    return twist


class PolicyRunnerNode(Node):
    def __init__(self):
        super().__init__('policy_runner_node')

        # Publisher for cmd_vel for a mobile robot.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create PolicyRunner once, using the existing model and logic.
        model_path = MODEL_PATH.resolve()
        self.get_logger().info(f'Loading policy from: {model_path}')
        self.runner = PolicyRunner(str(model_path))

        # Timer must run periodically and issue commands.
        self.create_timer(TIMER_PERIOD, self.timer_callback)

    def timer_callback(self):
        # Hardcoded test case for current step 0.
        robot_x, robot_y = 100.0, 100.0
        goal_x, goal_y = 400.0, 400.0

        dist = math.hypot(goal_x - robot_x, goal_y - robot_y)

        if dist < GOAL_THRESHOLD:
            # Goal reached: stop the robot.
            cmd = Twist()
            reason = 'at goal, stop'
        else:
            action = self.runner.predict_action(robot_x, robot_y, goal_x, goal_y)
            action_name = self.runner.action_name(action)
            cmd = action_to_twist(action)
            reason = f'action {action} ({action_name})'

        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f'obs=[{robot_x},{robot_y},{goal_x},{goal_y}] dist={dist:.2f} -> {reason}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyRunnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
