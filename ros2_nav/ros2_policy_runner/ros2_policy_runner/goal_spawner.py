import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from simulation_interfaces.srv import SpawnEntity, SetEntityState

from ros2_policy_runner.srv import SetGoal


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0


def green_box_sdf(name: str, size: float) -> str:
    return f"""
<sdf version='1.7'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <pose>0 0 {size/2.0} 0 0 0</pose>
      <collision name='collision'>
        <geometry>
          <box><size>{size} {size} {size}</size></box>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <box><size>{size} {size} {size}</size></box>
        </geometry>
        <material>
          <ambient>0 1 0 1</ambient>
          <diffuse>0 1 0 1</diffuse>
          <emissive>0 1 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class GoalSpawner(Node):
    def __init__(self):
        super().__init__("goal_spawner")

        self.declare_parameter("entity_name", "goal_block")
        self.declare_parameter("spawn_service", "/gzserver/spawn_entity")
        self.declare_parameter("pose_service", "/gzserver/set_entity_state")
        self.declare_parameter("goal_service", "/set_goal")
        self.declare_parameter("goal_threshold", 0.5)
        self.declare_parameter("bounds_min", 0.5)
        self.declare_parameter("bounds_max", 7.5)
        self.declare_parameter("box_size", 0.8)
        self.declare_parameter("box_z", 0.4)
        default_root = os.environ.get("RLNAV_ROOT", "")
        default_goal_sdf = ""
        if default_root:
            default_goal_sdf = str((Path(default_root) / "gz_sim" / "models_goal_block.sdf").resolve())
        self.declare_parameter("goal_sdf_path", default_goal_sdf)
        self.declare_parameter("spawn_retry_sec", 2.0)

        self.entity_name = self.get_parameter("entity_name").value
        self.spawn_service = self.get_parameter("spawn_service").value
        self.pose_service = self.get_parameter("pose_service").value
        self.goal_service = self.get_parameter("goal_service").value
        self.goal_threshold = float(self.get_parameter("goal_threshold").value)
        self.bounds_min = float(self.get_parameter("bounds_min").value)
        self.bounds_max = float(self.get_parameter("bounds_max").value)
        self.box_size = float(self.get_parameter("box_size").value)
        self.box_z = float(self.get_parameter("box_z").value)
        self.goal_sdf_path = str(self.get_parameter("goal_sdf_path").value)

        self.robot_pose = Pose2D()
        self.goal_pose = Pose2D()
        self.goal_initialized = False
        self.goal_active = False
        self.reach_latched = False
        self.have_odom = False
        self.last_move_time = self.get_clock().now()
        self.last_spawn_attempt = self.get_clock().now()
        self.spawn_retry_sec = float(self.get_parameter("spawn_retry_sec").value)
        self.spawned_entity = False
        self.last_goal_reason = "init"

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.spawn_client = self.create_client(SpawnEntity, self.spawn_service)
        self.pose_client = self.create_client(SetEntityState, self.pose_service)
        self.goal_client = self.create_client(SetGoal, self.goal_service)

        self._service_warned = False
        self._pending_goal = False
        self._spawn_inflight = False
        self.create_timer(0.2, self.timer_callback)

    def odom_callback(self, msg: Odometry):
        self.robot_pose.x = msg.pose.pose.position.x
        self.robot_pose.y = msg.pose.pose.position.y
        self.have_odom = True

    def timer_callback(self):
        if not self._ensure_spawn_pose_services():
            return

        if not self.goal_active:
            # Goal not active yet: ensure a single goal position and spawn the entity.
            if not self.goal_initialized:
                self._set_new_goal_position(reason="init")
            self._ensure_goal_entity()
            return

        # If goal service was down when we spawned, retry sending it.
        if self._pending_goal and self.goal_client.service_is_ready():
            self._send_goal_service()
            self._pending_goal = False

        if not self.have_odom:
            return

        dist = math.hypot(self.goal_pose.x - self.robot_pose.x, self.goal_pose.y - self.robot_pose.y)
        # Throttle distance logs (~1 Hz)
        if (self.get_clock().now() - self.last_move_time).nanoseconds >= 1_000_000_000:
            self.get_logger().info(
                f"robot=({self.robot_pose.x:.2f},{self.robot_pose.y:.2f}) goal=({self.goal_pose.x:.2f},{self.goal_pose.y:.2f}) dist={dist:.2f}"
            )
            self.last_move_time = self.get_clock().now()

        if dist <= self.goal_threshold and not self.reach_latched:
            self.reach_latched = True
            self.get_logger().info("Goal reached, relocating...")
            self._set_new_goal_position(reason="reached")
            self._move_goal_entity()
            return

        if dist > self.goal_threshold:
            self.reach_latched = False

    def _ensure_spawn_pose_services(self) -> bool:
        if self.spawn_client.service_is_ready() and self.pose_client.service_is_ready():
            return True

        # Auto-discover services if namespaced differently
        for name, types in self.get_service_names_and_types():
            if name.endswith("/spawn_entity") and "simulation_interfaces/srv/SpawnEntity" in types:
                if name != self.spawn_service:
                    self.spawn_service = name
                    self.spawn_client = self.create_client(SpawnEntity, self.spawn_service)
            if name.endswith("/set_entity_state") and "simulation_interfaces/srv/SetEntityState" in types:
                if name != self.pose_service:
                    self.pose_service = name
                    self.pose_client = self.create_client(SetEntityState, self.pose_service)

        if not self._service_warned:
            self.get_logger().info(
                f"Waiting for services: spawn={self.spawn_service}, pose={self.pose_service} ..."
            )
            self._service_warned = True
        return False

    def _spawn_new_goal(self, initial: bool):
        self.get_logger().info(
            f"Spawning goal at ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f}, {self.box_z:.2f})"
        )

        # First try to spawn; if it already exists, just move it.
        if initial and not self._spawn_inflight:
            req = SpawnEntity.Request()
            req.name = self.entity_name
            req.allow_renaming = False
            sdf_path = Path(self.goal_sdf_path)
            if sdf_path.is_file():
                # simulation_interfaces expects a URI for files
                req.uri = f"file://{sdf_path}"
            else:
                req.resource_string = green_box_sdf(self.entity_name, self.box_size)
                self.get_logger().info("Goal SDF file missing, using resource string")
            req.initial_pose = PoseStamped()
            req.initial_pose.header.frame_id = "world"
            req.initial_pose.pose.position.x = self.goal_pose.x
            req.initial_pose.pose.position.y = self.goal_pose.y
            req.initial_pose.pose.position.z = self.box_z
            self._spawn_inflight = True
            future = self.spawn_client.call_async(req)
            future.add_done_callback(self._spawn_done)
            # Also attempt a pose set in case the entity already exists.
            pose_req = SetEntityState.Request()
            pose_req.entity = self.entity_name
            pose_req.state.pose.position.x = self.goal_pose.x
            pose_req.state.pose.position.y = self.goal_pose.y
            pose_req.state.pose.position.z = self.box_z
            self.pose_client.call_async(pose_req)
        else:
            pose_req = SetEntityState.Request()
            pose_req.entity = self.entity_name
            pose_req.state.pose.position.x = self.goal_pose.x
            pose_req.state.pose.position.y = self.goal_pose.y
            pose_req.state.pose.position.z = self.box_z
            self.pose_client.call_async(pose_req)

        if self.goal_client.service_is_ready():
            self._send_goal_service()
            self._pending_goal = False
        else:
            self._pending_goal = True

        if initial:
            self.get_logger().info(
                f"Goal spawn requested name={self.entity_name} uri={req.uri or '<inline>'}"
            )
        else:
            self.goal_initialized = True
            self.goal_active = True
            self.get_logger().info(
                f"Goal set to ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f})"
            )

    def _spawn_done(self, future):
        self._spawn_inflight = False
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Goal spawn failed: {exc}")
            self.goal_initialized = False
            return

        res = getattr(result, "result", None)
        res_code = None
        res_msg = ""
        if res is not None:
            res_code = getattr(res, "result", None)
            res_msg = getattr(res, "error_message", "") or ""
        ok = (res_code == 1)
        msg = res_msg
        if ok or ("NAME_NOT_UNIQUE" in msg) or ("exists" in msg.lower()):
            self.spawned_entity = True
            self.goal_initialized = True
            self.goal_active = True
            self.get_logger().info(f"Goal spawned: {self.entity_name}")
        else:
            self.goal_initialized = False
            self.get_logger().error(f"Goal spawn error: {msg} (code={res_code})")
            self.spawned_entity = False

    def _set_new_goal_position(self, reason: str):
        old = (self.goal_pose.x, self.goal_pose.y)
        self.goal_pose.x = random.uniform(self.bounds_min, self.bounds_max)
        self.goal_pose.y = random.uniform(self.bounds_min, self.bounds_max)
        self.last_goal_reason = reason
        self.get_logger().info(
            f"New goal ({reason}) at ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f}, {self.box_z:.2f})"
        )
        if old != (0.0, 0.0) and reason != "reached":
            self.get_logger().warning(
                "Goal changed without reach event (reason=%s)" % reason
            )

    def _ensure_goal_entity(self):
        # Throttle spawn attempts
        if self._spawn_inflight:
            return
        now = self.get_clock().now()
        if (now - self.last_spawn_attempt).nanoseconds < int(self.spawn_retry_sec * 1e9):
            return
        self.last_spawn_attempt = now

        if not self.spawned_entity:
            self._spawn_new_goal(initial=True)
        else:
            # Entity exists; just move it to the current goal pose once.
            self._move_goal_entity()
            self.goal_active = True

    def _move_goal_entity(self):
        pose_req = SetEntityState.Request()
        pose_req.entity = self.entity_name
        pose_req.state.pose.position.x = self.goal_pose.x
        pose_req.state.pose.position.y = self.goal_pose.y
        pose_req.state.pose.position.z = self.box_z
        self.pose_client.call_async(pose_req)
        self.goal_active = True
        self._send_goal_service()
        self.get_logger().info(
            f"Goal moved to ({self.goal_pose.x:.2f}, {self.goal_pose.y:.2f})"
        )

    def _send_goal_service(self):
        goal_req = SetGoal.Request()
        goal_req.x = float(self.goal_pose.x)
        goal_req.y = float(self.goal_pose.y)
        self.goal_client.call_async(goal_req)


def main(args=None):
    rclpy.init(args=args)
    node = GoalSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
