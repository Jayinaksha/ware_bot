# ROS Workspace

This workspace contains the ROS 2 packages for the Inter IIT 2025 warehouse robotics challenge. It includes packages for robot simulation, navigation, scanning, and teleoperation.

## Packages

### `car`
- **Description**: Contains the robot description (URDF/Xacro) and simulation configuration.
- **Dependencies**: `joint_state_publisher`, `robot_state_publisher`, `gazebo_ros`, `xacro`, `ros_gz_bridge`.
- **Launch Files**:
  - `gazebo_model.launch.py`: Launches the robot model in the Gazebo simulation environment.

### `warehouse_navigation`
- **Description**: Implements the navigation stack for the warehouse robot, including path planning and execution.
- **Dependencies**: `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`, `nav2_msgs`, `action_msgs`, `scipy`, `numpy`, `custom_definitions`.
- **Python Package**: `navigation` (contains source code, actions, and data).

### `warehouse_scanning`
- **Description**: Handles shelf detection and scanning logic using robot sensors.
- **Dependencies**: `rclpy`, `nav_msgs`, `geometry_msgs`, `tf_transformations`, `custom_defintions`.
- **Python Package**: `shelf_detect` (contains source code).

### `r2d2_teleop`
- **Description**: Provides teleoperation capabilities for the robot using a joystick/gamepad.
- **Dependencies**: `rclpy`, `geometry_msgs`, `teleop_twist_joy`, `joy`.
- **Launch Files**:
  - `joystick.launch.py`: Launches the teleoperation nodes.

### `bno08x-ros2-driver`
- **Description**: ROS 2 Driver for the CEVA BNO08x IMU sensor.
- **Maintainer**: Balachandra Bhat.
- **Dependencies**: `rclcpp`, `sensor_msgs`, `std_msgs`.

### `ros2_laser_scan_matcher`
- **Description**: Provides laser scan matching functionality, likely for odometry or localization improvement.
- **Dependencies**: `nav2_common`, `sensor_msgs`, `csm`, `rclcpp`, `tf2`, etc.

### `csm`
- **Description**: A ROS wrapper for the Canonical Scan Matcher (CSM) library, used by `ros2_laser_scan_matcher`.
- **Details**: Pure C implementation of a fast ICP variation.

### `custom_definitions`
- **Description**: Defines custom ROS 2 messages and services used across the workspace.
- **Dependencies**: `rosidl_default_generators`, `rosidl_default_runtime`.

### `warehouse_robot_bringup`
- **Description**: Bringup package that launches the simulation, teleoperation, and navigation stack together.
- **Launch Files**:
  - `bringup.launch.py`: Launches `car/gazebo_model.launch.py`, `r2d2_teleop/joystick.launch.py`, and the `navigation` node.

## Build Instructions

To build the workspace, navigate to the `ros_workspace` directory and run:

```bash
colcon build
```

To build a specific package:

```bash
colcon build --packages-select <package_name>
```

## Setup

After building, source the setup file to add the packages to your environment:

```bash
source install/setup.bash
```

## Usage

1.  **Simulation**:
    ```bash
    ros2 launch car gazebo_model.launch.py
    ```

2.  **Teleoperation**:
    ```bash
    ros2 launch r2d2_teleop joystick.launch.py
    ```

3.  **Navigation & Scanning**:
    Ensure the simulation is running, then launch the respective nodes from `warehouse_navigation` or `warehouse_scanning` as needed (refer to package-specific documentation if available).

4.  **Full Bringup (Simulation, Teleop, Navigation)**:
    ```bash
    ros2 launch warehouse_robot_bringup bringup.launch.py
    ```
