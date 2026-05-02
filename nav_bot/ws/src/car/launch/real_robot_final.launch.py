import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, GroupAction, DeclareLaunchArgument, TimerAction, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    # --- PATHS ---
    pkg_car = get_package_share_directory('car')
    pkg_nav2 = get_package_share_directory('nav2_bringup')
    pkg_rplidar = get_package_share_directory('rplidar_ros')
    # pkg_imu = get_package_share_directory('bno08x_driver') # Uncomment if installed
    
    # --- CONFIG FILES ---
    ekf_params_file = os.path.join(pkg_car, 'config', 'ekf.yaml')
    slam_params_file = os.path.join(pkg_car, 'config', 'mapper_params_online_async.yaml')
    nav2_params_file = os.path.join(pkg_car, 'config', 'nav2_params.yaml')
    model_path = os.path.join(pkg_car, 'model', 'car.sdf') # Ensure this path is correct for your XACRO/URDF

    map_name = 'my_serial_map'
    map_folder = os.path.join(pkg_car, 'maps')
    serialized_map_path = os.path.join(map_folder, map_name)
    map_exists = os.path.exists(serialized_map_path + ".data")
    print(f"Checking for map at: {serialized_map_path}.data -> Exists? {map_exists}")
    # --- ARGS ---
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    )
    namespace = LaunchConfiguration('namespace')

    # ========================================================================
    # 1. HARDWARE DRIVERS
    # ========================================================================

    # A. Micro-ROS Agent (STM32)
    micro_ros_agent = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 
            'serial', '--dev', '/dev/ttyACM0', '-b', '115200'
        ],
        output='screen'
    )

    # B. RPLidar
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rplidar, 'launch', 'rplidar.launch.py')
        )
    )

    # C. BNO085 IMU (Example Node - Adjust executable if needed)
    imu_node = Node(
        package='bno08x_driver',
        executable='bno08x_driver_node',
        name='bno08x_driver',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    # ========================================================================
    # 2. STATE ESTIMATION & TF
    # ========================================================================

    # A. Robot State Publisher (REQUIRED for TFs: base_footprint -> sensors)
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': Command(['xacro ', model_path])}, 
            {'use_sim_time': False}
        ]
    )

    # B. Odom Publisher (STM32 Ticks -> /odom)
    odom_publisher = Node(
        package='car',
        executable='odom_publisher',
        name='odom_publisher',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    # C. Laser Filter (Clean scan for SLAM)
    laser_filter_node = Node(
        package='car',
        executable='simple_filter',
        name='simple_filter',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    # D. Laser Scan Matcher (Refines Odom using Lidar)
    scan_matcher = Node(
        package='ros2_laser_scan_matcher',
        executable='laser_scan_matcher',
        name='laser_scan_matcher',
        parameters=[{
            'base_frame': 'base_footprint',
            'odom_frame': 'odom',
            'laser_frame': 'laser',  # Must match your static TF
            'publish_odom': '/odom_laser',
            'publish_tf': False,     # False because EKF handles TF
            'use_imu': True,
            'use_odom': False,
            'max_correspondence_dist': 0.5,
            'max_iterations': 20,
            'use_sim_time': False
        }],
        remappings=[
            ('scan', '/scan_filtered'),
            ('imu', '/imu'), # Check if your BNO085 topic is /imu or /bno085/imu
            ('odom', '/odom_laser')
        ]
    )

    # D. EKF (Fuses Odom + IMU)
    localisation = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params_file, {'use_sim_time': False}]
    )

    # E. Static TF (Manual Lidar Offset)
    # CRITICAL: Parent frame must be 'base_footprint' to match your config
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # Adjust X/Z to match real mounting
        arguments=['0.1', '0', '0.15', '0', '0', '0', 'base_footprint', 'laser']
    )

    # ========================================================================
    # 3. MAPPING (SLAM)
    # ========================================================================
    start_mode = 'mapping'
    slam_params = [slam_params_file, {'use_sim_time': False}]
    if map_exists:
        slam_params.append({'map_file_name': serialized_map_path})
        print("--> LOADING EXISTING MAP FOR LIFELONG MAPPING")
    else:
        print("--> STARTING FRESH MAP")
        
    # A. SLAM Toolbox
    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file, {'use_sim_time': False}],
        remappings=[('scan', '/scan_filtered')]
    )

    # B. SLAM Lifecycle Manager (Prevents Map Resets)
    slam_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['slam_toolbox']},
            {'bond_timeout': 0.0} # Disable heartbeat to prevent resets
        ]
    )

    # ========================================================================
    # 4. NAVIGATION (Nav2)
    # ========================================================================
    
    nav2 = GroupAction([
        PushRosNamespace(namespace),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [pkg_nav2, 'launch', 'navigation_launch.py']
                )
            ),
            launch_arguments={
                'use_sim_time': 'False',
                'params_file': nav2_params_file
            }.items()
        ),
    ])

    # ========================================================================
    # 5. LOGIC & INTERFACE
    # ========================================================================

    # A. Hardware Interface (Stepper + QR Logic)
    hardware_interface = Node(
        package='shelf_detect',
        executable='hardware_interface',
        name='hardware_interface',
        output='screen'
    )

    # B. Visualizer (Shelf Detect)
    Visualise = Node(
        package='shelf_detect',
        executable='visualise',
        name='map_visualizer',
        output='screen'
    )
    
    # C. Navigation Node (The Mission Commander)
    Navigate = Node(
        package='navigation',
        executable='navigation',
        name='navigation_node',
        output='screen',
    )

    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================
    ld = LaunchDescription()
    
    # Args
    ld.add_action(namespace_arg)

    # Hardware & State
    ld.add_action(micro_ros_agent)
    ld.add_action(rplidar_launch)
    ld.add_action(imu_node)
    ld.add_action(nodeRobotStatePublisher)
    ld.add_action(odom_publisher)
    ld.add_action(lidar_tf)
    ld.add_action(laser_filter_node)
    ld.add_action(scan_matcher)
    
    # Localization & Mapping
    ld.add_action(localisation)
    ld.add_action(slam)
    ld.add_action(slam_lifecycle_manager)
    
    # Navigation
    ld.add_action(nav2)
    
    # Logic
    ld.add_action(hardware_interface)
    ld.add_action(Visualise)
    ld.add_action(Navigate)
    
    return ld