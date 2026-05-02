# Inter IIT 25 Project

## Project Overview
This project contains modules for QR code processing, rack detection, and a navigation bot using ROS.

## Directory Structure
- `QR processing/`: Contains scripts for QR code deblurring and decoding.
- `rack/`: Contains scripts for rack detection using YOLO and robust detection algorithms.
- `nav_bot/`: Contains the ROS workspace for the navigation bot.
- `ordered_invoices/`: Contains invoice PDFs.

## Installation

1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### QR Processing
Scripts for processing QR codes are located in `QR processing/`.
- `deepblind.py`: Deblurring script.
- `qr_decode.py`: QR code decoding script.

Example:
```bash
python "QR processing/qr_decode.py" --help
```

### Rack Detection
Scripts for rack detection are located in `rack/`.
- `robust_rack_detector.py`: Main detection script.

Example:
```bash
python "rack/robust_rack_detector.py" --help
```

### Navigation Bot
The `nav_bot` directory contains a ROS workspace.
1. Navigate to the workspace:
   ```bash
   cd nav_bot/ws
   ```
2. Build the workspace:
   ```bash
   colcon build
   ```
3. Source the setup script:
   ```bash
   source install/setup.bash
   ```

## Internal Structure Guidelines
(Pending specific guidelines from `Eternal_end_term_guidelines.pdf`)
