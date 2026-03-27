# Gazebo Sim Folder

This folder is a Gazebo (gz sim) harness for end-to-end RL inference tests.
It spawns the robot, a visible goal block, and runs the policy + base driver.

## Files

- `gz_sim/worlds/flat_world.sdf`: Simple flat world with ground plane and sun.
- `gz_sim/launch/hexa_sim.launch.py`: Launches gzserver + GUI and spawns the robot.
- `gz_sim/launch/hexa_agent.launch.py`: Full stack (sim + goal + policy + base driver).
- `gz_sim/models_goal_block.sdf`: Visible green goal entity.

## Run

From the repo root:

```bash
source /opt/ros/jazzy/setup.bash
source ./setup_env.sh
cd ros2_nav
colcon build --packages-select ros2_policy_runner --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
cd ..
ros2 launch gz_sim/launch/hexa_agent.launch.py use_gui:=True
```

## Important

This integration uses a **base-motion driver** and does **not** require joint-level control plugins.
You only need a valid URDF + meshes for the robot to appear.

## WSL GUI note

Gazebo GUI can crash on WSL due to GPU / EGL issues.
If you need a fallback, set software rendering explicitly:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
ros2 launch gz_sim/launch/hexa_agent.launch.py use_gui:=True
```
