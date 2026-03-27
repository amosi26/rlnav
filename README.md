# RL Navigation Project

This repository contains a simple RL navigation project organized into:

- `nav_core/`: environment, training script, inference wrapper, and policy tests
- `ros2_nav/`: minimal ROS2 package (`ros2_policy_runner`) that loads the trained PPO policy and publishes `/cmd_vel`
- `gz_sim/`: Gazebo Sim harness + launch files for end-to-end inference tests

## Key paths

- Trained model: `nav_core/models/finalmodel.zip` (not committed in public repo)
- Gym environment: `nav_core/gym_env.py`
- Training script: `nav_core/train_ppo_nav.py`
- Standalone inference test: `nav_core/test_inference.py`
- Policy behavior tests: `nav_core/run_policy_tests.py`
- ROS2 node: `ros2_nav/ros2_policy_runner/ros2_policy_runner/policy_node.py`

## Quick checks (from repo root)

```bash
python nav_core/test_inference.py
python nav_core/run_policy_tests.py
python -m compileall nav_core ros2_nav/ros2_policy_runner
```

## ROS2 run (from `ros2_nav/`)

```bash
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
colcon build --packages-select ros2_policy_runner
source install/setup.bash
ros2 run ros2_policy_runner policy_runner_node
```

## Gazebo inference (recommended)

This launches Gazebo, spawns the robot, spawns a visible goal entity, and runs the policy end-to-end.

```bash
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
source ./setup_env.sh
cd ros2_nav
colcon build --packages-select ros2_policy_runner --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
cd ..
ros2 launch gz_sim/launch/hexa_agent.launch.py use_gui:=True
```

### Notes
- The hexapod URDF + meshes are **not** included in the public repo. Provide your own in a local description package (see `gz_sim/README.md`).
- The trained model file is **not** included in the public repo. Set `RLNAV_MODEL_PATH` to your local model path if needed.
- The Gazebo integration uses **base motion only** (no joint control).
