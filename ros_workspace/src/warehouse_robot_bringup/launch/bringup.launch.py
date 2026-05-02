import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    car_pkg_dir = get_package_share_directory('car')
    teleop_pkg_dir = get_package_share_directory('r2d2_teleop')

    # Simulation launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(car_pkg_dir, 'launch', 'gazebo_model.launch.py')
        )
    )

    # Teleoperation launch
    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(teleop_pkg_dir, 'launch', 'joystick.launch.py')
        )
    )

    # Navigation Node
    navigation_node = Node(
        package='navigation',
        executable='navigation',
        name='navigation',
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        teleop_launch,
        navigation_node
    ])
