#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SimpleFilter(Node):
    def __init__(self):
        super().__init__('simple_filter')

        # QoS to match Gazebo (Best Effort)
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_policy)
            
        self.publisher = self.create_publisher(
            LaserScan, 
            '/scan_filtered', 
            qos_policy)

        # BLIND SPOT RADIUS (Meters)
        # Robot is ~0.21m from center to corner. 0.30m is safe.
        self.blind_spot_radius = 0.30 

    def listener_callback(self, msg):
        # Create a new message to publish
        clean_scan = msg
        
        # Filter the ranges
        # We replace any detection closer than 0.3m with 'infinity' (no obstacle)
        new_ranges = []
        for r in msg.ranges:
            if r < self.blind_spot_radius:
                new_ranges.append(float('inf'))
            else:
                new_ranges.append(r)
        
        clean_scan.ranges = new_ranges
        self.publisher.publish(clean_scan)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()