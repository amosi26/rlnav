#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"
source /opt/ros/jazzy/setup.bash
source ./setup_env.sh

cd "$ROOT_DIR/ros2_nav"
colcon build --packages-select ros2_policy_runner --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash

cd "$ROOT_DIR"
ros2 launch gz_sim/launch/hexa_agent.launch.py use_gui:=True
