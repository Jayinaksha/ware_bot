#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32

# --- CONFIGURATION ---
AXIS_LEFT_STICK_V = 1   # Forward/Back
AXIS_RIGHT_STICK_H = 3  # Turn
BUTTON_L1 = 4           # Enable Robot

# --- STEPPER TRIGGER MAPPING ---
# User Note: Default = 1.0 (Unpressed), Pressed = -1.0
AXIS_L2 = 2             # Stepper UP
AXIS_R2 = 5             # Stepper DOWN
TRIGGER_THRESHOLD = 0.5 # Must press trigger at least halfway (< 0.5) to activate
STEPPER_SPEED_VAL = 800 # Steps per second

# Speed Limits
MAX_LINEAR = 1.0 
MAX_ANGULAR = 1.5 

class SmartController(Node):
    def __init__(self):
        super().__init__('smart_controller')
        self.sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.step_pub = self.create_publisher(Int32, 'cmd_stepper', 10)
        
        self.active = False
        self.prev_btn = 0
        self.get_logger().info("🎮 READY: L1=Enable, L2=Lift UP, R2=Lift DOWN")

    def joy_callback(self, msg):
        # 1. Toggle Active (L1)
        if msg.buttons[BUTTON_L1] == 1 and self.prev_btn == 0:
            self.active = not self.active
            self.get_logger().info(f"State: {'🟢 ON' if self.active else '🔴 OFF'}")
        self.prev_btn = msg.buttons[BUTTON_L1]

        # 2. Drive Logic (Deadzone Included)
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        
        if self.active:
            raw_lin = msg.axes[AXIS_LEFT_STICK_V]
            raw_ang = msg.axes[AXIS_RIGHT_STICK_H]
            if abs(raw_lin) < 0.1: raw_lin = 0.0
            if abs(raw_ang) < 0.1: raw_ang = 0.0
            cmd.twist.linear.x = raw_lin * MAX_LINEAR
            cmd.twist.angular.z = raw_ang * MAX_ANGULAR
        self.cmd_pub.publish(cmd)

        # 3. Stepper Logic (Axis 2 & 5)
        # Default 1.0 -> Pressed -1.0. We check if < 0.5
        val_l2 = msg.axes[AXIS_L2]
        val_r2 = msg.axes[AXIS_R2]
        
        step_msg = Int32()
        
        if val_l2 < TRIGGER_THRESHOLD: 
            step_msg.data = STEPPER_SPEED_VAL  # UP
        elif val_r2 < TRIGGER_THRESHOLD:
            step_msg.data = -STEPPER_SPEED_VAL # DOWN
        else:
            step_msg.data = 0                  # STOP
            
        self.step_pub.publish(step_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SmartController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
