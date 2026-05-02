import os
import yaml
import tempfile
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Create a "Numbers Only" configuration dictionary
    # We remove 'box_frame' (String) and 'invert' (Bool) to appease the buggy parser.
    # The filter will default to using the LiDAR frame and invert=False.
    filter_config = {
        'scan_to_scan_filter_chain': {
            'ros__parameters': {
                'filter_chain': [
                    {
                        'name': 'box_filter',
                        'type': 'laser_filters/LaserScanBoxFilter',
                        'params': {
                            # ONLY FLOATS ALLOWED HERE
                            'max_x': 0.30,
                            'max_y': 0.25,
                            'max_z': 0.50,
                            'min_x': -0.30,
                            'min_y': -0.25,
                            'min_z': -0.50
                        }
                    }
                ]
            }
        }
    }

    # 2. Dump this dict to a temporary YAML file
    # This ensures the indentation is perfectly valid for ROS 2.
    temp_config_path = os.path.join(tempfile.gettempdir(), 'temp_laser_filter_config.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(filter_config, f, default_flow_style=False)

    # 3. Launch the Node using the temporary file
    return LaunchDescription([
        Node(
            package='laser_filters',
            executable='scan_to_scan_filter_chain',
            name='scan_to_scan_filter_chain', # Must match the YAML root
            output='screen',
            parameters=[temp_config_path],
            remappings=[
                ('scan', '/scan'),
                ('scan_filtered', '/scan_filtered')
            ],
            # Force Best Effort QoS to match Gazebo
            arguments=['--ros-args', '-p', 'qos_overrides./scan.subscription.reliability:=best_effort']
        )
    ])