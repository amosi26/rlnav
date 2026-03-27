cmake_minimum_required(VERSION 3.5)
project(ros2_policy_runner)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetGoal.srv"
)

install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/policy_runner_node
  scripts/gz_base_driver_node
  scripts/goal_spawner_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the node implementation into the generated Python package
install(FILES
  ros2_policy_runner/policy_node.py
  ros2_policy_runner/gz_base_driver.py
  ros2_policy_runner/goal_spawner.py
  DESTINATION ${PYTHON_INSTALL_DIR}/ros2_policy_runner
)

ament_package()
