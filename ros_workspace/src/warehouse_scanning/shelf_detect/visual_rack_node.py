#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import cv2
import os

# Import the logic from the file we just made
from .robust_rack_detector import RobustRackDetector

class VisualRackNode(Node):
    def __init__(self):
        super().__init__('visual_rack_node')

        # --- PARAMETER: Model Path ---
        # You can change this when running the node: ros2 run ... --ros-args -p model_path:=/new/path
        self.declare_parameter('model_path', '/home/noblehalogen/inter_iit25/nav_bot/ws/src/shelf_detect/shelf_detect/Rack_Detection_model_6.pt')
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value

        # --- SUBSCRIBERS ---
        # Listen to the robot camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # <--- CHECK THIS: Is your topic name correct?
            self.image_callback,
            10)
        
        # --- PUBLISHERS ---
        # 1. Alignment Error: Used to steer the robot (PID Control)
        self.error_pub = self.create_publisher(Float32, '/rack/alignment_error', 10)
        
        # 2. Status: Is a rack currently visible?
        self.visible_pub = self.create_publisher(Bool, '/rack/is_visible', 10)
        
        # 3. Debug Video: View this in RQT to see the Green/Red boxes
        self.debug_pub = self.create_publisher(Image, '/rack/visual_debug', 10)

        # --- INITIALIZATION ---
        self.bridge = CvBridge()
        self.get_logger().info(f"Loading Visual Detector Model: {self.model_path}")
        
        try:
            self.detector = RobustRackDetector(model_path=self.model_path)
            self.get_logger().info("Detector Ready!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.detector = None

    def image_callback(self, msg):
        if not self.detector: return

        try:
            # 1. Convert ROS Image -> OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 2. Run the Logic
            annotated_frame, result = self.detector.process_frame(frame)
            
            # 3. Publish Data
            self.visible_pub.publish(Bool(data=result.detected))
            
            if result.detected:
                self.error_pub.publish(Float32(data=result.center_error_x))
            
            # 4. Publish Debug Video
            debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualRackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()