#!/usr/bin/env python3
"""
predefined_path_navigator.py - Fixed & Working

UNIFIED NODE: Combines path following + proximity detection + pause control

Features:
- Reads path_record.csv (predefined strip waypoints)
- Sends waypoints sequentially to Nav2 /navigate_to_pose
- Subscribes to /detected_racks (shelves from detector)
- Checks proximity to detected shelves (threshold: 0.5m)
- Pauses 3 seconds when robot is near a shelf
- Resumes path following after pause
- Publishes status to /path_status for monitoring
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from custom_definitions.msg import RackArray
from std_msgs.msg import String

import csv
import math
import os
import time


CSV_PATH = 'path_record.csv'
SERVER_TIMEOUT = 5.0
PROXIMITY_THRESHOLD = 0.5  # meters
PAUSE_DURATION = 3.0  # seconds


class PredefinedPathNavigator(Node):

    def __init__(self):
        super().__init__('predefined_path_navigator')

        # Parameters
        self.declare_parameter('csv_path', CSV_PATH)
        self.declare_parameter('goal_timeout_sec', 120.0)
        self.declare_parameter('proximity_threshold', PROXIMITY_THRESHOLD)
        self.declare_parameter('pause_duration', PAUSE_DURATION)

        self.csv_path = self.get_parameter('csv_path').value
        self.goal_timeout = self.get_parameter('goal_timeout_sec').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.pause_duration = self.get_parameter('pause_duration').value

        # Load path from CSV
        self.path_points = []
        self.load_path(self.csv_path)

        if not self.path_points:
            self.get_logger().error("No path points loaded!")
            return

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 50)
        self.racks_sub = self.create_subscription(
            RackArray, '/detected_racks', self.racks_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/path_status', 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # State: Path Following
        self.current_point_index = 0
        self.active_goal_handle = None
        self.active_goal_result_future = None
        self.goal_start_time = None
        self.goal_sent = False

        # State: Proximity & Pause
        self.robot_pose = None
        self.detected_racks = []
        self.pausing = False
        self.pause_start_time = None
        self.currently_paused_racks = set()
        self.pause_goal_handle = None

        # Main loop timer
        self.timer = self.create_timer(0.5, self.main_loop)

        self.get_logger().info("=" * 70)
        self.get_logger().info("PREDEFINED PATH NAVIGATOR (UNIFIED)")
        self.get_logger().info(f"Loaded {len(self.path_points)} path points from CSV")
        self.get_logger().info(f"Proximity threshold: {self.proximity_threshold}m")
        self.get_logger().info(f"Pause duration: {self.pause_duration}s")
        self.get_logger().info(f"Goal timeout: {self.goal_timeout}s")
        self.get_logger().info("=" * 70)

    # ===== CSV LOADING =====

    def load_path(self, csv_path):
        """Load path from CSV (x, y, yaw_deg)."""
        if not os.path.exists(csv_path):
            self.get_logger().error(f"CSV not found: {csv_path}")
            return

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x = float(row['x'])
                        y = float(row['y'])
                        yaw_deg = float(row['yaw_deg'])
                        self.path_points.append((x, y, yaw_deg))
                    except Exception as e:
                        self.get_logger().warn(f"Skipping bad row {row}: {e}")
        except Exception as e:
            self.get_logger().error(f"Failed to read CSV: {e}")

    # ===== SUBSCRIPTIONS =====

    def odom_callback(self, msg: Odometry):
        """Track robot pose from filtered odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_pose = (pos.x, pos.y, ori.x, ori.y, ori.z, ori.w)

    def racks_callback(self, msg: RackArray):
        """Update detected rack positions."""
        self.detected_racks = []
        for rack in msg.racks:
            self.detected_racks.append((rack.x, rack.y))

    # ===== MAIN STATE MACHINE =====

    def main_loop(self):
        """Main periodic loop that orchestrates path following + pause logic."""
        try:
            # Priority 1: If pausing, handle pause duration
            if self.pausing:
                self.handle_pause()
                return

            # Priority 2: Check proximity to shelves (may trigger pause)
            self.check_proximity_and_pause()

            # Priority 3: Path following state machine
            self.path_following()

        except Exception as e:
            self.get_logger().error(f"Exception in main_loop: {e}")

    # ===== PROXIMITY DETECTION & PAUSE TRIGGER =====

    def check_proximity_and_pause(self):
        """Check if robot is near any detected shelf, trigger pause if so."""
        if self.robot_pose is None or not self.detected_racks:
            return

        robot_x, robot_y, qx, qy, qz, qw = self.robot_pose

        for i, (rack_x, rack_y) in enumerate(self.detected_racks):
            dist = math.hypot(rack_x - robot_x, rack_y - robot_y)

            if dist < self.proximity_threshold:
                # Near this rack
                if i not in self.currently_paused_racks:
                    self.get_logger().info(
                        f"✓ SHELF DETECTED at distance {dist:.3f}m - TRIGGERING PAUSE"
                    )
                    self.currently_paused_racks.add(i)
                    self.trigger_pause(robot_x, robot_y, qx, qy, qz, qw)

        # Clean up racks we've moved away from
        racks_to_remove = []
        for rack_id in self.currently_paused_racks:
            if rack_id < len(self.detected_racks):
                rack_x, rack_y = self.detected_racks[rack_id]
                dist = math.hypot(rack_x - robot_x, rack_y - robot_y)
                if dist >= self.proximity_threshold:
                    racks_to_remove.append(rack_id)

        for rack_id in racks_to_remove:
            self.currently_paused_racks.discard(rack_id)

    def trigger_pause(self, x, y, qx, qy, qz, qw):
        """Initiate pause at current location."""
        self.pausing = True
        self.pause_start_time = time.time()

        # Send a goal at current position to stop robot
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.pause_goal_response_cb)

        self.publish_status(f"PAUSING near shelf for {self.pause_duration}s")

    def pause_goal_response_cb(self, future):
        """Callback for pause goal response from Nav2."""
        try:
            handle = future.result()
            if handle.accepted:
                self.pause_goal_handle = handle
                self.get_logger().info("⏸ Pause goal accepted by Nav2")
            else:
                self.get_logger().warn("Pause goal rejected by Nav2")
        except Exception as e:
            self.get_logger().error(f"Error in pause_goal_response_cb: {e}")

    def handle_pause(self):
        """Check pause duration; resume when done."""
        elapsed = time.time() - self.pause_start_time
        remaining = self.pause_duration - elapsed

        if remaining > 0:
            self.publish_status(f"PAUSING...{remaining:.1f}s remaining")
        else:
            self.get_logger().info(f"✓ Pause complete ({elapsed:.1f}s). Resuming path...")

            # Cancel pause goal so robot can resume
            if self.pause_goal_handle is not None:
                try:
                    self.pause_goal_handle.cancel_goal_async()
                except:
                    pass
                self.pause_goal_handle = None

            self.pausing = False
            self.pause_start_time = None
            self.publish_status("RESUMING path")

    # ===== PATH FOLLOWING =====

    def path_following(self):
        """Handle path following state machine."""
        # Check for stale goal (async race condition fix)
        if self.active_goal_handle is not None and self.active_goal_result_future is not None:
            if self.active_goal_result_future.done():
                # Goal finished
                try:
                    result = self.active_goal_result_future.result()
                    status = result.status
                    if status == GoalStatus.STATUS_SUCCEEDED:
                        self.get_logger().info(f"✓ Waypoint {self.current_point_index + 1} REACHED")
                        self.current_point_index += 1
                    else:
                        self.get_logger().warn(f"✗ Goal failed (status={status}), retrying...")
                except Exception as e:
                    self.get_logger().error(f"Error getting result: {e}")

                self.active_goal_handle = None
                self.active_goal_result_future = None
                self.goal_sent = False
                self.goal_start_time = None
                return

        # Check for timeout
        if self.goal_start_time is not None:
            elapsed = time.time() - self.goal_start_time
            if elapsed > self.goal_timeout:
                self.get_logger().warn(f"Goal timeout ({elapsed:.1f}s), cancelling...")
                try:
                    self.active_goal_handle.cancel_goal_async()
                except:
                    pass
                self.active_goal_handle = None
                self.active_goal_result_future = None
                self.goal_sent = False
                self.goal_start_time = None
                return

        # All points done?
        if self.current_point_index >= len(self.path_points):
            self.get_logger().info("=" * 70)
            self.get_logger().info("✓✓✓ PATH COMPLETE! ✓✓✓")
            self.get_logger().info("=" * 70)
            self.publish_status("PATH COMPLETE")
            return

        # Goal already sent for current point?
        if self.goal_sent:
            return

        # Send goal for current path point
        self.send_goal_to_nav2()

    def send_goal_to_nav2(self):
        """Send current path point as goal to Nav2."""
        point = self.path_points[self.current_point_index]
        x, y, yaw_deg = point

        self.get_logger().info(
            f"→ Path point {self.current_point_index + 1}/{len(self.path_points)}: "
            f"x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}°"
        )

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation = self.yaw_to_quat(math.radians(yaw_deg))

        if not self.nav_client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
            self.get_logger().error("Nav2 server not available")
            return

        self.get_logger().info("→ Sending to Nav2...")
        self.goal_sent = True
        self.goal_start_time = time.time()

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        """Nav2 accepted or rejected goal."""
        try:
            handle = future.result()
            if not handle.accepted:
                self.get_logger().warn("Goal REJECTED by Nav2")
                self.goal_sent = False
            else:
                self.get_logger().info("Goal ACCEPTED by Nav2, waiting for completion...")
                self.active_goal_handle = handle
                result_fut = handle.get_result_async()
                self.active_goal_result_future = result_fut
        except Exception as e:
            self.get_logger().error(f"Error in goal_response_cb: {e}")
            self.goal_sent = False

    # ===== UTILITIES =====

    def publish_status(self, text: str):
        """Publish status message."""
        msg = String()
        msg.data = text
        try:
            self.status_pub.publish(msg)
        except:
            pass

    @staticmethod
    def yaw_to_quat(yaw):
        """Convert yaw (radians) to Quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.z = sy
        q.w = cy
        return q


def main(args=None):
    rclpy.init(args=args)
    node = PredefinedPathNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
