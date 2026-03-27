from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    root = Path(__file__).resolve().parent

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(root / "hexa_sim.launch.py"))
    )

    base_driver = Node(
        package="ros2_policy_runner",
        executable="gz_base_driver_node",
        output="screen",
    )

    goal_spawner = Node(
        package="ros2_policy_runner",
        executable="goal_spawner_node",
        output="screen",
    )

    policy_node = Node(
        package="ros2_policy_runner",
        executable="policy_runner_node",
        output="screen",
        parameters=[{
            "auto_randomize_goal": False,
            "randomize_robot_start": False,
        }],
    )

    return LaunchDescription([
        sim_launch,
        base_driver,
        goal_spawner,
        policy_node,
    ])
