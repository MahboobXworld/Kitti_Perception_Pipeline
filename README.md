# KITTI 3D Sensor Fusion Perception Pipeline

[![ROS Version](https://img.shields.io/badge/ROS-Noetic%20%2F%20Melodic-blue.svg)](http://wiki.ros.org/)
[![Python Version](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Ultralytics-YOLOv8-red.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-KITTI-orange.svg)](http://www.cvlibs.net/datasets/kitti/)

A ROS-based, real-time 3D sensor fusion pipeline that combines 2D camera images with 3D LiDAR point clouds to perform robust object detection, depth estimation, and 3D camera-frame localization. Optimized for the **KITTI Dataset**, this package integrates deep learning-based object detection (`YOLOv8`) with geometric camera calibration projections.

---

## 🌟 Key Features

*   **Real-time 2D Object Detection & Tracking:** Utilizes `Ultralytics YOLOv8` (`yolov8n.pt`) with optional `ByteTrack` multi-object tracking.
*   **LiDAR Cloud Filtering:** Real-time PointCloud2 processing, featuring NaN/Inf cleanup, 3D Region of Interest (ROI) filtering, ground suppression, and downsampling (Voxel Grid & Random).
*   **Flexible Camera Projection:** Supports pinhole and fisheye camera models, Plumb Bob distortion correction, homogeneous projection matrices, and camera rectification.
*   **Z-Buffer Occlusion Filtering:** Discards background points projected onto foreground object pixels, maintaining geometric integrity.
*   **Vectorized Data Association:** High-performance Numpy-based point-in-bounding-box association to calculate median object depth and physical 3D camera-frame coordinates.
*   **Diagnostic & Visual Feedback:** Publishes custom ROS messages containing 3D object attributes, alongside dynamic color-coded depth overlays on the camera feed.

---

## 📂 Project Structure

```text
perception_pipeline/
├── CMakeLists.txt              # Catkin build configuration
├── package.xml                 # ROS package dependencies and metadata
├── setup.py                    # Python package setup for workspace importing
├── config/                     # Configuration parameters
│   ├── fusion.yaml             # Node topics, sync, and visualization settings
│   ├── lidar_filter.yaml       # Point cloud crop and downsample settings
│   ├── projection.yaml         # Extrinsics, intrinsics, rectification matrices
│   ├── yolo_detector.yaml      # YOLOv8 inference and tracking parameters
│   └── visualize.rviz          # Configured Rviz visualization layout
├── launch/
│   └── pipeline.launch         # Launch file (plays ROS Bag, loads parameters, starts nodes)
├── msg/                        # Custom ROS message definitions
│   ├── Object.msg              # Bounding box, class, confidence, depth, and 3D centroid
│   └── Objects.msg             # Header and list of Object messages
├── scripts/
│   └── fusion_node.py          # Main ROS node (synchronizer and fusion manager)
└── src/perception_pipeline/    # Reusable Python modules
    ├── __init__.py
    ├── lidar_filter.py         # Point cloud filtering and downsampling
    ├── projection_module.py    # Calibration transforms and pixel projections
    └── yolo_detector.py        # YOLOv8 wrapper (preprocess, infer, postprocess)
```

---

## 🛠️ Prerequisites & Installation

### 1. ROS Dependencies
Ensure you have a ROS Melodic or Noetic installation (Ubuntu 18.04 / 20.04). You will need the following ROS packages:
```bash
sudo apt-get install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-message-filters ros-$ROS_DISTRO-tf ros-$ROS_DISTRO-sensor-msgs ros-$ROS_DISTRO-geometry-msgs
```

### 2. Python Packages
Install the required Python packages in your workspace environment:
```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  # For GPU (recommended)
# Or for CPU-only:
# pip3 install torch torchvision torchaudio
pip3 install ultralytics pyyaml numpy opencv-python
```

### 3. Build the Package
Clone this repository into your Catkin workspace source directory (e.g., `~/catkin_ws/src`):
```bash
cd ~/catkin_ws/src
git clone https://github.com/<your-username>/perception_pipeline.git
cd ~/catkin_ws
catkin_make
# Or using catkin build:
# catkin build perception_pipeline
```
*Note: Make sure to source your workspace setup file in each new terminal:*
```bash
source devel/setup.bash
```

---

## 🚀 How to Run the Pipeline

### 1. Configure the ROS Bag Path
The launch file is configured to play a KITTI bag file. By default, it looks for the dataset bag at:
`/home/mahboob-alam/perception_ws/src/datasets/kitti_full_0047.bag`

If your dataset bag is located elsewhere, update the path in `launch/pipeline.launch` (Line 11):
```xml
<node pkg="rosbag" type="play" name="player" output="screen" args="--clock /path/to/your/kitti_dataset.bag" />
```

### 2. Launch the Pipeline
Run the launch file to start the dataset player, parameters server, fusion node, and RViz:
```bash
roslaunch perception_pipeline pipeline.launch
```

---

## ⚙️ Configuration Parameters

The pipeline is highly configurable through YAML files in the `config/` directory.

### Fusion (`config/fusion.yaml`)
*   `subscribe`: Input image and lidar topic names.
*   `sync`: Sensor message synchronization queue size and slop (maximum time difference, e.g. `0.03`s for 30Hz).
*   `visualization`: Enable/disable depth overlays and select color maps (e.g. `HSV`, `JET`, `VIRIDIS`).

### LiDAR Filter (`config/lidar_filter.yaml`)
*   `x_min`/`x_max`/`y_min`/`y_max`/`z_min`/`z_max`: Defines the 3D bounding box for points of interest.
*   `min_range`/`max_range`: Limits points by radial Euclidean distance.
*   `ground_filter`: Suppresses road points below `ground_z`.
*   `downsample`: Enable voxelization (`leaf_size`) or random sampling (`percentage`) to optimize CPU usage.

### Projection (`config/projection.yaml`)
*   `intrinsics`: Camera focal length ($f_x$, $f_y$) and optical center ($c_x$, $c_y$).
*   `resolution`: Target image dimensions `[width, height]`.
*   `projection_matrix`: $3 \times 4$ projection matrix ($P_2$ for KITTI Camera 2).
*   `rectification_matrix`: $4 \times 4$ camera rectification matrix ($R_{rect}$).
*   `extrinsic/transform_matrix`: $4 \times 4$ rigid transform matrix ($T_{velo}^{cam}$).

### YOLO Detector (`config/yolo_detector.yaml`)
*   `model_path`: Relative or absolute path to the YOLO checkpoint (e.g., `models/yolov8n.pt`).
*   `device`: Inference hardware (`cpu` or `cuda:0`).
*   `tracking`: Enable ByteTrack tracking (`true` / `false`).
*   `allowed_classes`: Classes to process (e.g. `car`, `person`, `bicycle`, etc.).

---

## 📡 Custom Messages

The pipeline publishes custom messages of type `perception_pipeline/Objects` on `/fusion/objects`.

### `perception_pipeline/Object.msg`
Represents a single fused 3D detection:
```text
string class_name                # Class classification (e.g., "car")
float32 confidence               # Detector confidence score [0.0 - 1.0]
int32 id                         # Tracking ID (if tracking is enabled; else -1)
float32 depth                    # Calculated median depth in meters
geometry_msgs/Point position     # Physical 3D centroid in camera frame [X, Y, Z]
sensor_msgs/RegionOfInterest bbox # 2D Bounding Box in image coordinates
```

### `perception_pipeline/Objects.msg`
Represents a collection of detections in a frame:
```text
std_msgs/Header header
perception_pipeline/Object[] objects
```
