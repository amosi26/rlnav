from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from ros_gz_sim.actions import GzServer
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import random


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent
    urdf_path = LaunchConfiguration("urdf_path")
    world_path = LaunchConfiguration("world_path")
    robot_name = LaunchConfiguration("robot_name")
    world_name = LaunchConfiguration("world_name")
    world_sdf = LaunchConfiguration("world_sdf")
    use_gui = LaunchConfiguration("use_gui")

    default_world = str(root / "worlds" / "flat_world.sdf")

    return LaunchDescription([
        DeclareLaunchArgument(
            "urdf_path",
            default_value=str(repo_root / "hexa_description" / "urdf" / "robot.urdf"),
            description="Path to URDF to spawn",
        ),
        DeclareLaunchArgument(
            "world_path",
            default_value=str(root / "worlds" / "flat_world.sdf"),
            description="Path to world SDF",
        ),
        DeclareLaunchArgument(
            "robot_name",
            default_value="hexa",
            description="Name of the spawned robot",
        ),
        DeclareLaunchArgument(
            "spawn_x",
            default_value=str(random.uniform(0.5, 6.5)),
            description="Spawn X position",
        ),
        DeclareLaunchArgument(
            "spawn_y",
            default_value=str(random.uniform(0.5, 6.5)),
            description="Spawn Y position",
        ),
        DeclareLaunchArgument(
            "spawn_z",
            default_value="0.2",
            description="Spawn Z position",
        ),
        DeclareLaunchArgument(
            "world_sdf",
            default_value=TextSubstitution(text=default_world),
            description="Path to the SDF world file",
        ),
        DeclareLaunchArgument(
            "use_gui",
            default_value=TextSubstitution(text="False"),
            description="Launch gz GUI (WSL may crash)",
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="flat_world",
            description="Gazebo world name (for spawn)",
        ),
        GzServer(
            world_sdf_file=world_sdf,
        ),
        ExecuteProcess(
            cmd=["gz", "sim", "-g"],
            output="screen",
            condition=IfCondition(use_gui),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(Path(get_package_share_directory("ros_gz_sim")) / "launch" / "gz_spawn_model.launch.py")
            ),
            launch_arguments={
                "world": world_name,
                "file": urdf_path,
                "entity_name": robot_name,
                "x": LaunchConfiguration("spawn_x"),
                "y": LaunchConfiguration("spawn_y"),
                "z": LaunchConfiguration("spawn_z"),
                "allow_renaming": "false",
            }.items(),
        ),
    ])
