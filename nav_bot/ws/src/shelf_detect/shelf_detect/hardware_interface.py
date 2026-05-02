#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from std_srvs.srv import Trigger
import cv2
import time
from ultralytics import YOLO

class HardwareInterfaceNode(Node):
    def __init__(self):
        super().__init__('hardware_interface_node')
        
        # Use ReentrantCallbackGroup so the Service can run in parallel with the Camera Timer
        self.cb_group = ReentrantCallbackGroup()

        # --- PUBLISHERS ---
        # Talk to STM32 Micro-ROS
        self.stepper_pub = self.create_publisher(Vector3, '/stepper_cmd', 10)
        # Publish QR data for logging/debugging
        self.qr_pub = self.create_publisher(String, '/qr_data', 10)
        
        # --- SERVICES ---
        # Navigation calls this service when it reaches a rack
        self.scan_srv = self.create_service(
            Trigger, 
            '/perform_rack_scan', 
            self.scan_sequence_callback, 
            callback_group=self.cb_group
        )
        
        # --- CAMERA SETUP ---
        try:
            self.model = YOLO("qr_model.pt") 
            self.cap = cv2.VideoCapture(0) # Camera Index 0
            if not self.cap.isOpened():
                self.get_logger().error("Could not open Camera!")
        except Exception as e:
            self.get_logger().error(f"Error loading YOLO/Camera: {e}")
        
        # Run QR detection continuously
        self.timer = self.create_timer(0.1, self.qr_timer_callback, callback_group=self.cb_group)
        
        self.get_logger().info("Hardware Interface (Stepper + QR) Ready!")

    # --- QR DETECTION LOOP ---
    def qr_timer_callback(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened(): return

        ret, frame = self.cap.read()
        if not ret: return

        # Run inference
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        for result in results:
            for box in result.boxes:
                # Assuming class 0 is QR_Code
                if int(box.cls[0]) == 0: 
                    msg = String()
                    msg.data = "QR_DETECTED"
                    self.qr_pub.publish(msg)
                    # Log to console so we know it worked
                    self.get_logger().info(">>> QR CODE DETECTED! <<<")

    # --- STEPPER SEQUENCE LOGIC ---
    def scan_sequence_callback(self, request, response):
        self.get_logger().info("Received Scan Request. Starting Sequence...")
        
        msg = Vector3()
        msg.y = 1.0 # Enable Motor Driver (ENA_PIN)

        # 1. MOVE UP
        self.get_logger().info("Stepper: MOVING UP")
        msg.x = 1.0 
        self.stepper_pub.publish(msg)
        time.sleep(4.0) # Adjust duration based on rack height

        # 2. HOLD & SCAN
        self.get_logger().info("Stepper: HOLDING (Scanning...)")
        msg.x = 0.0 # Stop moving
        self.stepper_pub.publish(msg)
        time.sleep(3.0) # Give camera time to see the QR code

        # 3. MOVE DOWN
        self.get_logger().info("Stepper: MOVING DOWN")
        msg.x = -1.0 
        self.stepper_pub.publish(msg)
        time.sleep(4.0) # Adjust to match UP time

        # 4. DISABLE
        self.get_logger().info("Stepper: DONE. Disabling.")
        msg.x = 0.0
        msg.y = 0.0 # Disable Motor Driver
        self.stepper_pub.publish(msg)

        response.success = True
        response.message = "Scan Sequence Completed Successfully"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterfaceNode()
    
    # CRITICAL: Use MultiThreadedExecutor so Timer (QR) and Service (Stepper) run at same time
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()