import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import math
from rclpy.time import Time

class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')
        
        # Configuration matches your STM32 constants
        self.ticks_per_rev = 600.0
        self.wheel_diameter = 0.15
        self.wheel_separation = 0.40 # Distance between left and right wheels (Estimate)
        self.meters_per_tick = (math.pi * self.wheel_diameter) / self.ticks_per_rev

        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        self.prev_left_count = 0
        self.prev_right_count = 0
        self.last_time = self.get_clock().now()

        # Subscribe to STM32 encoder data
        self.sub = self.create_subscription(Vector3, 'encoder_count', self.enc_callback, 10)
        
        # Publish standard Odometry for Nav2
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def enc_callback(self, msg):
        current_time = self.get_clock().now()
        
        # 1. Read raw ticks
        left_count = msg.x
        right_count = msg.y

        # 2. Calculate change
        d_left = (left_count - self.prev_left_count) * self.meters_per_tick
        d_right = (right_count - self.prev_right_count) * self.meters_per_tick
        
        self.prev_left_count = left_count
        self.prev_right_count = right_count

        # 3. Calculate distance and rotation
        d_center = (d_left + d_right) / 2.0
        d_phi = (d_right - d_left) / self.wheel_separation

        # 4. Update Pose (Simple Euler integration)
        self.x += d_center * math.cos(self.th)
        self.y += d_center * math.sin(self.th)
        self.th += d_phi

        # 5. Create Quaternion for orientation
        q_z = math.sin(self.th / 2.0)
        q_w = math.cos(self.th / 2.0)

        # 6. Publish TF (Odom -> Base_Link)
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = q_z
        t.transform.rotation.w = q_w
        self.tf_broadcaster.sendTransform(t)

        # 7. Publish Odom Message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = q_z
        odom.pose.pose.orientation.w = q_w
        self.odom_pub.publish(odom)

def main():
    rclpy.init()
    node = OdomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()