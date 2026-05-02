import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        # 1. Driver for the Controller (Reads hardware)
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{'coalesce_interval': 0.02}],
        ),

        # 2. Your Custom Controller (Handles Toggle + Mapping + TwistStamped)
        Node(
            package='r2d2_teleop',
            executable='cmd_vel_converter', # We reuse the same entry point name
            name='smart_controller',
            output='screen'
        )
    ])
