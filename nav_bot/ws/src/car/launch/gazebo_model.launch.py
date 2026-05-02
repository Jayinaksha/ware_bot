import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
import xacro

def generate_launch_description():
    
    # 1. Package and Robot Names
    namePackage = 'car'
    robotXacroName = 'car'
    
    # 2. Get Dynamic Paths (Fixes the "anany" vs "noblehalogen" error)
    pkg_path = get_package_share_directory(namePackage)
    modelFileRelativePath = 'model/car.sdf'
    pathModelFile = os.path.join(pkg_path, modelFileRelativePath)
    
    default_world = os.path.join(pkg_path, 'worlds', 'warehouse.sdf')
    
    
    # These point to YOUR config folder automatically
    slam_params_file = os.path.join(pkg_path, 'config', 'mapper_params_online_async.yaml')
    ekf_params_file = os.path.join(pkg_path, 'config', 'ekf.yaml')
    nav2_params_file = os.path.join(pkg_path, 'config', 'nav2_params.yaml')
    laser_filter_config = os.path.join(pkg_path, 'config', 'laser_filter.yaml')
    
    # 3. Launch Arguments
    world = LaunchConfiguration('world')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='World to load'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    )
    namespace = LaunchConfiguration('namespace')

    # 4. Gazebo Launch
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    )
    
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPackageLaunch, 
        launch_arguments={'gz_args': ['-r -v1 ', world], 'on_exit_shutdown':"true"}.items()
    )
    
    # 5. Nodes
    
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', pathModelFile])}, {'use_sim_time': True}]
    )
    
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robotXacroName,
            '-x', '-8.0', '-y', '-3.0', '-z', '0.4', '-Y', '1.57079632679',
        ],
        output='screen',
    )

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
           "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
        ],
        output='screen',
        parameters=[{'use_sim_time':True}]
    )
    
    ros_gz_image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image_raw"]
    )

    laser_filter_node = Node(
        package='shelf_detect',
        executable='simple_filter',
        name='simple_filter',
        output='screen'
    )

    scan_matcher_node = Node(
        package='ros2_laser_scan_matcher',
        executable='laser_scan_matcher',
        name='laser_scan_matcher',
        parameters=[{
            'base_frame': 'base_footprint',
            'odom_frame': 'odom',
            'laser_frame': 'lidar_link',
            'publish_odom': '/odom_laser', # Publish topic
            'publish_tf': False,            # Publish TF (odom -> base_footprint)
            'use_imu': True,
            'use_odom': False,
            'max_correspondence_dist': 2.0,
            'max_iterations': 50,
            'kf_dist_linear': 0.1,  # Update every 10cm
            'kf_dist_angular': 0.1, # Update every ~5 degrees             # Ignore wheels
            'max_iterations': 10, # Try 50 times to align before giving up
            'outliers_maxPerc': 0.80, # Keep 90% of points even if they look wrong
            'sigma': 0.010
        }],
        remappings=[
            ('scan', '/scan_filtered'),
            ('imu', '/imu'),
            ('odom', '/odom_laser')   # IMPORTANT: Use filtered scan
        ],
        arguments=['--ros-args', '-p', 'qos_overrides./scan.subscription.reliability:=best_effort']
    )
    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file, {'use_sim_time': True}],
        remappings=[('scan', '/scan_filtered')],
        arguments=['--ros-args', '-p', 'qos_overrides./scan.subscription.reliability:=best_effort']    
    )

    slam_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['slam_toolbox']},
            {'bond_timeout': 0.0}
        ]
    )
    
    # THIS WAS THE BROKEN PART: Now it uses the correct path variable
    localisation = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params_file, {'use_sim_time': True}],
    )
    
    nav2 = GroupAction([
        PushRosNamespace(namespace),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py']
                )
            ),
            launch_arguments={
                'use_sim_time': 'True',
                'use_robot_state_pub': 'False',
                'use_composition': 'False',
                'log_level': 'error',
                'params_file': nav2_params_file # Uses correct path
            }.items()
        ),
    ])

    Visualise = Node(
        package='shelf_detect',
        executable='visualise',
        name='map_visualizer',
        output='screen'
    )
    
    Navigate = Node(
        package='navigation',
        executable='navigation',
        name='navigation_node',
        output='screen',
    )
    
    # 6. Launch Description Assembly
    ld = LaunchDescription()
    
    # Setup Arguments FIRST
    ld.add_action(world_arg)
    ld.add_action(namespace_arg)
    
    # Then Launch Gazebo
    ld.add_action(gazeboLaunch)
    
    # Then Bridges and Robot
    ld.add_action(start_gazebo_ros_bridge_cmd)
    ld.add_action(ros_gz_image_bridge)
    ld.add_action(nodeRobotStatePublisher)
    ld.add_action(spawnModelNodeGazebo)
    
    # Then Navigation Stack
   
    ld.add_action(laser_filter_node)
    ld.add_action(scan_matcher_node)
    ld.add_action(localisation)
    ld.add_action(slam)
    ld.add_action(slam_lifecycle_manager)
    ld.add_action(nav2)
    
    # Then Logic
    ld.add_action(Visualise)
    ld.add_action(Navigate)
    
    return ld