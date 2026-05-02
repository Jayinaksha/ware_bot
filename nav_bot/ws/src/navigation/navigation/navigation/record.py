#!/usr/bin/env python3
"""
record_path.py

Record the robot path while you drive manually (e.g. with teleop).

- Subscribes to /odometry/filtered (change to /odom if needed)
- Every time the robot has moved more than MIN_DIST between samples,
  it stores (x, y, yaw_deg)
- On shutdown (Ctrl+C), it writes all samples to a CSV file.

Usage:
  1) Start your full stack (Gazebo + EKF + teleop)
  2) In another terminal:
       ros2 run <your_package> record_path
  3) Drive the robot along the desired strip with teleop
  4) Ctrl+C the node
  5) Check the saved CSV (default: path_record.csv in current directory)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import csv
import os
from rclpy.executors import ExternalShutdownException

MIN_DIST = 0.25  # meters between recorded samples
OUTPUT_CSV = 'path_record.csv'


class PathRecorder(Node):
    def __init__(self):
        super().__init__('path_recorder')

        # Parameter: choose odometry topic
        self.declare_parameter('odom_topic', '/odometry/filtered')
        odom_topic = self.get_parameter('odom_topic').value

        self.subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            50
        )

        self.path_points = []  # list of (x, y, yaw_deg)
        self.last_x = None
        self.last_y = None

        self.get_logger().info('=' * 70)
        self.get_logger().info('PATH RECORDER STARTED')
        self.get_logger().info(f'Subscribing to odometry: {odom_topic}')
        self.get_logger().info(f'Min distance between samples: {MIN_DIST} m')
        self.get_logger().info(f'Will save to: {os.path.abspath(OUTPUT_CSV)}')
        self.get_logger().info('Drive with teleop, then Ctrl+C to save.')
        self.get_logger().info('=' * 70)

    def odom_callback(self, msg: Odometry):
        # Current position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Orientation yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        yaw_deg = math.degrees(yaw)

        if self.last_x is None:
            # First sample
            self.last_x = x
            self.last_y = y
            self.path_points.append((x, y, yaw_deg))
            self.get_logger().info(f'First point: x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}°')
            return

        dist = math.hypot(x - self.last_x, y - self.last_y)
        if dist >= MIN_DIST:
            self.path_points.append((x, y, yaw_deg))
            self.last_x = x
            self.last_y = y
            self.get_logger().info(f'Recorded point #{len(self.path_points)}: '
                                   f'x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}°, '
                                   f'dist_from_prev={dist:.3f} m')

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        """Convert quaternion to yaw (radians)."""
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def save_to_csv(self):
        if not self.path_points:
            self.get_logger().warn('No path points recorded, nothing to save.')
            return

        try:
            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'yaw_deg'])
                for x, y, yaw_deg in self.path_points:
                    writer.writerow([x, y, yaw_deg])
            self.get_logger().info(f'Saved {len(self.path_points)} points to {os.path.abspath(OUTPUT_CSV)}')
        except Exception as e:
            self.get_logger().error(f'Failed to save CSV: {e}')

    def destroy_node(self):
        # Override to write on shutdown
        self.get_logger().info('Shutting down PathRecorder, saving path...')
        self.save_to_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Keyboard interrupt / external shutdown.')
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()