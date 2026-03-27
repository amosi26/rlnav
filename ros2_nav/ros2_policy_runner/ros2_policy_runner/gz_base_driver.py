import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from simulation_interfaces.srv import SetEntityState


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def yaw_to_quaternion(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class GzBaseDriver(Node):
    def __init__(self):
        super().__init__("gz_base_driver")

        self.declare_parameter("entity_name", "hexa")
        self.declare_parameter("service_name", "/gzserver/set_entity_state")
        self.declare_parameter("timer_period", 0.05)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("bounds_min", 0.0)
        self.declare_parameter("bounds_max", 8.0)

        self.entity_name = self.get_parameter("entity_name").value
        self.service_name = self.get_parameter("service_name").value
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.bounds_min = float(self.get_parameter("bounds_min").value)
        self.bounds_max = float(self.get_parameter("bounds_max").value)

        self.pose = Pose2D()
        self.cmd = Twist()
        self.last_time = self.get_clock().now()
        self._last_log_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)

        self.client = self.create_client(SetEntityState, self.service_name)
        self._service_ready = False
        self._service_warned = False

        period = float(self.get_parameter("timer_period").value)
        self.timer = self.create_timer(period, self.timer_callback)
        self.get_logger().info(
            f"Driving entity '{self.entity_name}' via {self.service_name}"
        )

    def cmd_callback(self, msg: Twist):
        self.cmd = msg

    def timer_callback(self):
        if not self._ensure_service():
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            return

        vx = float(self.cmd.linear.x)
        vy = float(self.cmd.linear.y)

        # Direct planar motion in world frame (matches training env)
        self.pose.x += vx * dt
        self.pose.y += vy * dt
        self.pose.x = max(self.bounds_min, min(self.pose.x, self.bounds_max))
        self.pose.y = max(self.bounds_min, min(self.pose.y, self.bounds_max))

        req = SetEntityState.Request()
        req.entity = self.entity_name
        req.state.pose.position.x = self.pose.x
        req.state.pose.position.y = self.pose.y
        req.state.pose.position.z = 0.2
        qx, qy, qz, qw = yaw_to_quaternion(0.0)
        req.state.pose.orientation.x = qx
        req.state.pose.orientation.y = qy
        req.state.pose.orientation.z = qz
        req.state.pose.orientation.w = qw
        self.client.call_async(req)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = self.pose.x
        odom.pose.pose.position.y = self.pose.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        self.odom_pub.publish(odom)

        # Throttled status log (~1 Hz)
        if (now - self._last_log_time).nanoseconds >= 1_000_000_000:
            self._last_log_time = now
            self.get_logger().info(
                f"pose=({self.pose.x:.2f},{self.pose.y:.2f}) cmd=(vx={vx:.2f}, vy={vy:.2f})"
            )

    def _ensure_service(self) -> bool:
        if self._service_ready and self.client.service_is_ready():
            return True

        if self.client.service_is_ready():
            self._service_ready = True
            if self._service_warned:
                self.get_logger().info(f"Service available: {self.service_name}")
            return True

        # Try auto-discovery for any /set_entity_pose service.
        for name, types in self.get_service_names_and_types():
            if name.endswith("/set_entity_state") and "simulation_interfaces/srv/SetEntityState" in types:
                if name != self.service_name:
                    self.get_logger().info(
                        f"Switching service to {name}"
                    )
                    self.service_name = name
                    self.client = self.create_client(SetEntityState, self.service_name)
                if self.client.service_is_ready():
                    self._service_ready = True
                    return True

        if not self._service_warned:
            self.get_logger().info(
                f"Waiting for service {self.service_name} ..."
            )
            self._service_warned = True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = GzBaseDriver()
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
