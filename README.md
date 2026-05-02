# Inter IIT 25 — Warehouse Bot

## Project Overview

An autonomous warehouse robot system for Inter IIT Tech Meet 25. The project combines deep-learning-based QR deblurring, YOLO + Gabor-filter rack detection, and a ROS 2 navigation stack that drives the robot around a warehouse, scans each rack with a stepper-mounted camera, and reads QR codes on every shelf item.

---

## Repository Structure

```
ware_bot/
├── QR processing/          # Standalone image deblurring & QR decoding scripts
│   ├── deepblind.py        # DeepBlind blind-deblurring (U-Net, single image or folder)
│   ├── deepgyro.py         # DeepGyro gyro-aided deblurring
│   ├── models.py           # U-Net architectures for DeepBlind and DeepGyro
│   ├── calibration.py      # Camera intrinsics & IMU calibration parameters
│   ├── IO.py               # IMU data loader, image reader/writer, folder utils
│   ├── utils.py            # Spatial/temporal alignment, blur-field computation, quaternion integration
│   ├── generate.py         # Orchestrates IMU + camera data to generate blur fields
│   ├── visualize.py        # Plots blur vectors overlaid on images
│   └── qr_decode.py        # OpenCV QR detector; validates RACKID_SHELFID_ITEMCODE format
│
├── rack/                   # Standalone rack detection scripts
│   ├── robust_rack_detector.py       # YOLO + Gabor-filter pipeline (main, robot-ready)
│   ├── Rack_Detection_Balanced.py    # Batch processor: YOLO + Canny edge validation + NMS
│   ├── test_dataset.py               # Batch test runner using RobustRackDetector + tqdm
│   ├── manual_tuning_script.py       # Interactive dataset tester (mirrors test_dataset.py)
│   └── Rack_project/                 # YOLO training project assets
│
├── nav_bot/ws/             # ROS 2 (Humble) colcon workspace
│   └── src/
│       ├── car/            # Robot URDF/XACRO model, Gazebo worlds, launch files
│       ├── shelf_detect/   # Perception + hardware ROS 2 package (Python)
│       ├── navigation/     # Autonomous navigation state-machine package (Python)
│       ├── custom_definitions/  # Custom ROS 2 message definitions (e.g. RackArray)
│       ├── csm/            # Laser scan matcher dependency
│       └── ros2_laser_scan_matcher/
│
├── requirements.txt        # Python dependencies for standalone scripts
└── Eternal_end_term_guidelines.pdf
```

---

## Installation

### Standalone scripts (QR processing / Rack detection)

```bash
# Clone the repo
git clone <repo-url>
cd ware_bot

# Install Python dependencies
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `matplotlib`, `Pillow`, `keras`, `tensorflow`, `opencv-python`, `ultralytics`, `tqdm`

### ROS 2 workspace

Requires **ROS 2 Humble** and **Nav2** installed on Ubuntu 22.04.

```bash
cd nav_bot/ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

---

## Usage

### QR Processing

#### `qr_decode.py` — Detect & decode a QR code from an image

```bash
python "QR processing/qr_decode.py" --image <path/to/image.jpg> --output qr_result.jpg
```

Decodes the QR code, draws a bounding box, and validates the expected warehouse format:
`RACKID_SHELFID_ITEMCODE` (e.g. `R03_S2_ITM430`).

#### `deepblind.py` — Blind image deblurring (no IMU required)

```bash
python "QR processing/deepblind.py" -i <input_image_or_folder> -o <output_folder> -w DeepBlind.hdf5
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Input image file or folder |
| `-o` | `output` | Output folder |
| `-w` | `DeepBlind.hdf5` | Pre-trained model weights |

#### `deepgyro.py` — Gyro-aided deblurring

Reads `blurred/`, `blurx/`, and `blury/` sub-folders from the input path and writes deblurred images using the `DeepGyro.hdf5` checkpoint.

```bash
python "QR processing/deepgyro.py" -i <dataset_folder>
```

#### `generate.py` — Generate blur fields from IMU + camera data

Loads camera calibration from `calibration.py` and IMU data via `IO.py`, computes rotation (quaternion integration) and linear motion, then writes blur field images to `blurred/`, `blurx/`, `blury/`, and `visualization/` sub-folders.

---

### Rack Detection

#### `robust_rack_detector.py` — Single-image test (YOLO + Gabor)

```bash
python rack/robust_rack_detector.py --image <path/to/image.jpg> \
                                     --model yolo11n.pt \
                                     --output output.jpg
```

The `RobustRackDetector` class:
1. Runs YOLO inference at 40 % confidence.
2. Validates each bounding box with a Gabor filter bank (4 orientations) to confirm wire-mesh texture.
3. Returns a `RackResult` with `detected`, `box`, `center_error_x` (−1.0 … +1.0), and `distance_factor`.

#### `Rack_Detection_Balanced.py` — Batch processing with edge validation

Hard-code `TEST_IMAGES_FOLDER` and `YOLO_MODEL_PATH` at the top of the file, then run:

```bash
python rack/Rack_Detection_Balanced.py
```

Outputs are written to a timestamped folder: `valid_detections/`, `rejected_images/`, `edge_maps/`, and `SUMMARY_REPORT.txt`.

#### `test_dataset.py` — Batch test with progress bar

```bash
python rack/test_dataset.py --input <images_folder> \
                             --output <results_folder> \
                             --model yolo11n.pt
```

---

### Navigation Bot (ROS 2)

#### Simulation (Gazebo)

```bash
cd nav_bot/ws
source install/setup.bash
ros2 launch car gazebo_model.launch.py
```

#### Real Robot

```bash
ros2 launch car real_robot_final.launch.py
```

#### Key ROS 2 nodes

| Node | Package | Description |
|------|---------|-------------|
| `visual_rack_node` | `shelf_detect` | Subscribes `/camera/image_raw`, publishes rack alignment error on `/rack/alignment_error` and visibility on `/rack/is_visible` |
| `qr_detector_node` | `shelf_detect` | Continuous QR detection from camera; publishes detections on `/qr_data` |
| `hardware_interface` | `shelf_detect` | Exposes `/perform_rack_scan` service; controls stepper motor via `/stepper_cmd` (STM32 Micro-ROS) and detects QR codes |
| `odom_publisher` | `shelf_detect` | Converts STM32 encoder ticks (`/encoder_count`) to Nav2 `Odometry` and TF |
| `navigation_node` | `navigation` | Full state-machine: SLAM → align → enter warehouse → visit racks → scan → exit |
| `map_autosaver` | `shelf_detect` | Periodically saves the SLAM map to disk |

#### Navigation state machine

```
IDLE → slam_completed → reaching_origin → entering_warehouse
     → navigating_to_racks123 → intermediate_point_bw_123_45
     → navigating_to_racks45 → exiting_warehouse
```

The mission is triggered by calling the `start_mission` service:

```bash
ros2 service call /start_mission std_srvs/srv/Trigger {}
```

#### ROS 2 topics at a glance

| Topic | Type | Direction |
|-------|------|-----------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input |
| `/rack/alignment_error` | `std_msgs/Float32` | Output (−1.0 … +1.0) |
| `/rack/is_visible` | `std_msgs/Bool` | Output |
| `/rack/visual_debug` | `sensor_msgs/Image` | Output (annotated video) |
| `/qr_data` | `std_msgs/String` | Output |
| `/stepper_cmd` | `geometry_msgs/Vector3` | Output to STM32 |
| `/encoder_count` | `geometry_msgs/Vector3` | Input from STM32 |
| `/detected_racks` | `custom_definitions/RackArray` | Input to navigation |
| `/map` | `nav_msgs/OccupancyGrid` | Input from SLAM Toolbox |
| `/odometry/filtered` | `nav_msgs/Odometry` | Input from EKF |
