#!/usr/bin/env python3
"""
predefined_path_navigator_bt.py

UNIFIED NODE with BEHAVIOR TREE:
- Service trigger `/start_navigation` (SetBool) to enable/disable
- Behavior Tree coordinates:
  * Path following with Nav2
  * Proximity shelf detection + scanning behavior (replaces pause)
- Shelf scanning with rotation and dwell time
- YOLO detection integration during scanning
- Image capture and JSON data logging
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.clock import Clock

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import Image

from custom_definitions.msg import RackArray
from std_msgs.msg import String
from std_srvs.srv import SetBool

import py_trees
import py_trees.console as console
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Fallback, Parallel, ReactiveSequence
from py_trees.decorators import Timeout
from py_trees.common import Status

import csv
import math
import os
import time
import json
from datetime import datetime


CSV_PATH = '/home/rudy/ws/src/navigation/data/path_record.csv'
SCAN_LOG_PATH = '/home/rudy/ws/src/navigation/data/scan_detections.json'
SERVER_TIMEOUT = 5.0
PROXIMITY_THRESHOLD = 0.5
SCAN_DURATION = 3.0
ROTATION_SPEED = 0.5  # rad/s for scanning
ROTATION_INCREMENT = 15  # degrees per scan step


class PredefinedPathNavigator(Node):
    """Main ROS2 node managing navigation and behavior tree."""

    def __init__(self):
        super().__init__('navigation')
        self.get_logger().info("NODE CONSTRUCTOR STARTED")

        # Parameters
        self.declare_parameter('csv_path', CSV_PATH)
        self.declare_parameter('goal_timeout_sec', 120.0)
        self.declare_parameter('proximity_threshold', PROXIMITY_THRESHOLD)
        self.declare_parameter('scan_duration', SCAN_DURATION)

        self.csv_path = self.get_parameter('csv_path').value
        self.goal_timeout = self.get_parameter('goal_timeout_sec').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.scan_duration = self.get_parameter('scan_duration').value

        # Load path
        self.path_points = []
        self.load_path(self.csv_path)

        if not self.path_points:
            self.get_logger().error("No path points loaded!")
            self.path_points = [] 
            

        # State variables
        self.navigation_enabled = False  # SERVICE CONTROLLED
        self.robot_pose = None
        self.detected_racks = []
        self.current_point_index = 0
        self.active_goal_handle = None
        self.active_goal_result_future = None
        self.goal_sent = False
        self.goal_start_time = None
        
        self.scan_images = []
        self.scan_detections = []
        self.current_scan_rotation = 0.0

        # Subscriptions
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 50)
        self.create_subscription(RackArray, '/detected_racks', self.racks_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/path_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # Create and setup behavior tree
        self.behaviour_tree = self.create_behavior_tree()
        self.behaviour_tree.setup(timeout=15)

        # Timer for BT ticking (faster than main loop)
        self.tree_timer = self.create_timer(0.1, self.tick_tree)

        # Service server for start/stop
        self.srv = self.create_service(SetBool, '/start_navigation', self.handle_start_nav)

        self.get_logger().info("=" * 70)
        self.get_logger().info("PREDEFINED PATH NAVIGATOR WITH BEHAVIOR TREE")
        self.get_logger().info("Call:")
        self.get_logger().info("  ros2 service call /start_navigation std_srvs/srv/SetBool '{data: true}'")
        self.get_logger().info("=" * 70)

    # =============================
    #    SERVICE CALLBACK
    # =============================
    def handle_start_nav(self, request, response):
        """Enable/disable navigation via service."""
        if request.data:
            if not self.navigation_enabled:
                self.navigation_enabled = True
                self.current_point_index = 0
                self.get_logger().info("Navigation ENABLED")
                self.publish_status("NAVIGATION ENABLED")
            response.success = True
            response.message = "Navigation started"
        else:
            self.navigation_enabled = False
            self.get_logger().info("Navigation DISABLED — Cancelling goals")
            self.publish_status("NAVIGATION DISABLED")

            # Cancel any active Nav2 goal
            if self.active_goal_handle:
                try:
                    self.active_goal_handle.cancel_goal_async()
                except Exception as e:
                    self.get_logger().warn(f"Failed to cancel goal: {e}")

            self.active_goal_handle = None
            self.active_goal_result_future = None
            self.goal_sent = False

            # Stop rotation if scanning
            self.stop_rotation()

            response.success = True
            response.message = "Navigation stopped"

        return response

    # =============================
    #    BEHAVIOR TREE CREATION
    # =============================
    def create_behavior_tree(self):
        """
        Build behavior tree structure:
        
        Root (ReactiveSequence)
        ├── NavigationSequence
        │   ├── CheckNavigationEnabled
        │   ├── CheckNextWaypoint
        │   └── FollowPathBranch
        │       ├── SendNav2Goal
        │       └── WaitForGoalComplete
        └── ShelfScanningSequence
            ├── CheckShelfProximity
            ├── CancelNav2Goal
            └── ScanShelfBehavior
        """
        root = ReactiveSequence(name="RootBehavior")

        # Navigation branch
        nav_sequence = Sequence(name="NavigationSequence", memory=False)
        nav_sequence.add_children([
            CheckNavigationEnabled(self, name="IsNavEnabled"),
            CheckNextWaypoint(self, name="HasNextWaypoint"),
            SendNav2Goal(self, name="SendGoal"),
            WaitForGoalComplete(self, name="WaitGoalDone"),
        ])

        # Shelf scanning branch
        scan_sequence = Sequence(name="ShelfScanningSequence", memory=False)
        scan_sequence.add_children([
            CheckNavigationEnabled(self, name="IsNavEnabled2"),
            CheckShelfProximity(self, name="NearShelf"),
            CancelNav2Goal(self, name="StopNav2"),
            ScanShelfBehavior(self, name="ScanShelf"),
        ])

        root.add_children([nav_sequence, scan_sequence])
        return root

    def tick_tree(self):
        """Tick the behavior tree at regular intervals."""
        if not self.navigation_enabled:
            return
        
        try:
            self.behaviour_tree.tick_once()
        except Exception as e:
            self.get_logger().error(f"BT tick error: {e}")

    # =============================
    #    CSV LOADING
    # =============================
    def load_path(self, csv_path):
        """Load waypoints from CSV file."""
        if not os.path.exists(csv_path):
            self.get_logger().error(f"CSV not found: {csv_path}")
            return

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = float(row['x'])
                    y = float(row['y'])
                    yaw_deg = float(row['yaw_deg'])
                    self.path_points.append((x, y, yaw_deg))
            self.get_logger().info(f"Loaded {len(self.path_points)} waypoints from CSV")
        except Exception as e:
            self.get_logger().error(f"Failed to read CSV: {e}")

    # =============================
    #    SUBSCRIPTIONS
    # =============================
    def odom_callback(self, msg):
        """Update robot pose from odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_pose = (pos.x, pos.y, ori.x, ori.y, ori.z, ori.w)

    def racks_callback(self, msg):
        """Update detected racks list."""
        self.detected_racks = [(rack.x, rack.y) for rack in msg.racks]

    def image_callback(self, msg):
        """Capture images during scanning (optional YOLO integration)."""
        if len(self.scan_images) < 10:
            self.scan_images.append({
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'frame_id': msg.header.frame_id,
                'height': msg.height,
                'width': msg.width
            })

    # =============================
    #    NAV2 HELPERS
    # =============================
    def send_next_goal(self):
        """Send next waypoint to Nav2."""
        if self.current_point_index >= len(self.path_points):
            self.get_logger().info("PATH COMPLETE")
            self.publish_status("PATH COMPLETE")
            return False

        x, y, yaw_deg = self.path_points[self.current_point_index]

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation = self.yaw_to_quat(math.radians(yaw_deg))

        if not self.nav_client.wait_for_server(SERVER_TIMEOUT):
            self.get_logger().error("Nav2 not available")
            return False

        self.get_logger().info(
            f"Sending waypoint {self.current_point_index + 1} / {len(self.path_points)}"
        )

        self.goal_sent = True
        self.goal_start_time = time.time()

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_cb)
        return True

    def goal_response_cb(self, future):
        """Callback when Nav2 accepts/rejects goal."""
        try:
            handle = future.result()
            if not handle.accepted:
                self.get_logger().warn("Goal rejected by Nav2")
                self.goal_sent = False
            else:
                self.active_goal_handle = handle
                self.active_goal_result_future = handle.get_result_async()
        except Exception as e:
            self.get_logger().error(f"Goal response error: {e}")
            self.goal_sent = False

    def check_goal_complete(self):
        """Check if current Nav2 goal is complete."""
        if not self.active_goal_result_future or not self.active_goal_result_future.done():
            return None

        try:
            result = self.active_goal_result_future.result()
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"Reached waypoint {self.current_point_index + 1}")
                self.current_point_index += 1
                success = True
            else:
                self.get_logger().warn(f"Goal failed with status {result.status}")
                success = False
        except Exception as e:
            self.get_logger().error(f"Goal result error: {e}")
            success = False

        self.active_goal_handle = None
        self.active_goal_result_future = None
        self.goal_sent = False
        self.goal_start_time = None

        return success

    def check_goal_timeout(self):
        """Check if goal has timed out."""
        if self.goal_start_time and time.time() - self.goal_start_time > self.goal_timeout:
            self.get_logger().warn("Goal timeout")
            try:
                if self.active_goal_handle:
                    self.active_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f"Failed to cancel timeout goal: {e}")

            self.active_goal_handle = None
            self.active_goal_result_future = None
            self.goal_sent = False
            self.goal_start_time = None
            return True
        return False

    def cancel_nav2_goal(self):
        """Cancel current Nav2 goal."""
        if self.active_goal_handle:
            try:
                self.active_goal_handle.cancel_goal_async()
                self.get_logger().info("Nav2 goal cancelled")
            except Exception as e:
                self.get_logger().warn(f"Failed to cancel goal: {e}")

        self.active_goal_handle = None
        self.active_goal_result_future = None
        self.goal_sent = False
        self.goal_start_time = None

    # =============================
    #    ROTATION & SCANNING
    # =============================
    def start_rotation(self, angular_velocity=ROTATION_SPEED):
        """Start rotating robot (for scanning)."""
        twist = Twist()
        twist.angular.z = angular_velocity
        self.cmd_vel_pub.publish(twist)

    def stop_rotation(self):
        """Stop robot rotation."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def save_scan_data(self):
        """Save scan images and detections to JSON log."""
        if not self.scan_images:
            return
        
        scan_entry = {
            'timestamp': datetime.now().isoformat(),
            'waypoint_index': self.current_point_index,
            'robot_pose': {
                'x': self.robot_pose[0],
                'y': self.robot_pose[1]
            } if self.robot_pose else None,
            'detected_racks': self.detected_racks,
            'image_count': len(self.scan_images),
            'rotation_angle_deg': self.current_scan_rotation,
            'custom_detections': self.scan_detections
        }
        
        try:
            existing_data = []
            if os.path.exists(SCAN_LOG_PATH):
                with open(SCAN_LOG_PATH, 'r') as f:
                    existing_data = json.load(f)
            
            existing_data.append(scan_entry)
            
            with open(SCAN_LOG_PATH, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            self.get_logger().info(f"Scan data saved: {len(self.scan_images)} frames")
        except Exception as e:
            self.get_logger().error(f"Failed to save scan data: {e}")
        
        self.scan_images = []
        self.scan_detections = []

    # =============================
    #    UTILITIES
    # =============================
    def publish_status(self, text):
        """Publish status message."""
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    @staticmethod
    def yaw_to_quat(yaw):
        """Convert yaw angle to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.z = sy
        q.w = cy
        return q


# =============================
#    BEHAVIOR TREE NODES
# =============================

class CheckNavigationEnabled(Behaviour):
    """Check if navigation is enabled via service."""

    def __init__(self, navigator, name="CheckNavEnabled"):
        super().__init__(name=name)
        self.navigator = navigator

    def update(self):
        if self.navigator.navigation_enabled:
            return Status.SUCCESS
        return Status.FAILURE


class CheckNextWaypoint(Behaviour):
    """Check if there's a next waypoint to follow."""

    def __init__(self, navigator, name="CheckNextWaypoint"):
        super().__init__(name=name)
        self.navigator = navigator

    def update(self):
        if self.navigator.current_point_index < len(self.navigator.path_points):
            return Status.SUCCESS
        return Status.FAILURE


class SendNav2Goal(Behaviour):
    """Send next waypoint goal to Nav2."""

    def __init__(self, navigator, name="SendNav2Goal"):
        super().__init__(name=name)
        self.navigator = navigator

    def update(self):
        # Only send if not already sent
        if self.navigator.goal_sent:
            return Status.RUNNING

        # Try to send goal
        if self.navigator.send_next_goal():
            return Status.RUNNING
        return Status.FAILURE


class WaitForGoalComplete(Behaviour):
    """Wait for Nav2 goal completion and handle timeouts."""

    def __init__(self, navigator, name="WaitGoalComplete"):
        super().__init__(name=name)
        self.navigator = navigator

    def update(self):
        # Check timeout
        if self.navigator.check_goal_timeout():
            return Status.FAILURE

        # Check if goal completed
        result = self.navigator.check_goal_complete()
        if result is not None:
            return Status.SUCCESS if result else Status.FAILURE

        # Still waiting
        return Status.RUNNING


class CheckShelfProximity(Behaviour):
    """Check if any shelf is within proximity threshold."""

    def __init__(self, navigator, name="CheckShelfProximity"):
        super().__init__(name=name)
        self.navigator = navigator

    def update(self):
        if self.navigator.robot_pose is None or not self.navigator.detected_racks:
            return Status.FAILURE

        robot_x, robot_y = self.navigator.robot_pose[0:2]

        for rack_x, rack_y in self.navigator.detected_racks:
            dist = math.hypot(rack_x - robot_x, rack_y - robot_y)
            if dist < self.navigator.proximity_threshold:
                self.navigator.publish_status(
                    f"SHELF DETECTED at {dist:.2f}m — SCANNING"
                )
                return Status.SUCCESS

        return Status.FAILURE


class CancelNav2Goal(Behaviour):
    """Cancel current Nav2 goal."""

    def __init__(self, navigator, name="CancelNav2Goal"):
        super().__init__(name=name)
        self.navigator = navigator

    def update(self):
        self.navigator.cancel_nav2_goal()
        return Status.SUCCESS


class ScanShelfBehavior(Behaviour):
    """Scan detected shelf: rotate and dwell with data logging."""

    def __init__(self, navigator, name="ScanShelf"):
        super().__init__(name=name)
        self.navigator = navigator
        self.scan_start_time = None
        self.total_rotation = 0.0

    def initialise(self):
        """Called when behavior is first ticked."""
        self.scan_start_time = time.time()
        self.total_rotation = 0.0
        self.navigator.start_rotation(ROTATION_SPEED)
        self.navigator.scan_images = []
        self.navigator.scan_detections = []
        self.navigator.publish_status(f"SCANNING SHELF for {self.navigator.scan_duration}s")

    def update(self):
        """Check if scan duration elapsed and update rotation tracking."""
        if self.scan_start_time is None:
            return Status.FAILURE

        elapsed = time.time() - self.scan_start_time
        remaining = self.navigator.scan_duration - elapsed
        
        # Calculate rotation angle (0.5 rad/s * elapsed time)
        self.total_rotation = ROTATION_SPEED * elapsed * (180 / math.pi)
        self.navigator.current_scan_rotation = self.total_rotation

        if remaining > 0:
            # Update status with remaining time and rotation angle
            self.navigator.publish_status(
                f"SCANNING... {remaining:.1f}s | Rotated {self.total_rotation:.1f}°"
            )
            return Status.RUNNING

        # Scan complete - save data
        self.navigator.stop_rotation()
        self.navigator.save_scan_data()
        self.navigator.publish_status("SCAN COMPLETE — Resuming navigation")
        return Status.SUCCESS

    def terminate(self, new_status):
        """Called when behavior is halted."""
        self.navigator.stop_rotation()
        if new_status != Status.SUCCESS:
            self.navigator.save_scan_data()
        self.scan_start_time = None


# =============================
#    MAIN ENTRY POINT
# =============================

def main(args=None):
    rclpy.init(args=args)
    node = PredefinedPathNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()