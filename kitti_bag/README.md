# KITTI Raw Dataset to ROS Bag Converter

This utility converts raw **KITTI** recordings (camera frames, Velodyne point clouds, GPS/IMU data) into a single ROS `.bag` file. The output bag contains synchronized image, lidar, and inertial topics suitable for replaying inside a ROS workspace.

---

## 🛠️ Prerequisites

Ensure you have the required ROS 1 packages (like `rosbag` and `cv_bridge`) installed and sourced. Additionally, install the following Python packages:

```bash
pip3 install pykitti opencv-python numpy
```

---

## 📥 Downloading the KITTI Dataset

1. Go to the [KITTI Raw Data website](http://www.cvlibs.net/datasets/kitti/raw_data.php).
2. Choose any sequence/drive (e.g. from the "Road" or "City" sections).
3. Download two key archives for the chosen drive:
    *   **[synced+rectified data]** (contains Velodyne, cameras, and oxts directories).
    *   **[calibration]** (contains the rigid-body and camera calibration text files).

---

## 📁 Directory Structure Organization

The `pykitti` parser requires a strict folder layout to link the calibration files with the sensor recordings. Extract your downloaded zip files so that they match the following structure:

```text
datasets/                      <--- This is your --base_path
└── <date>/                    <--- e.g., 2011_09_26
    ├── calib_cam_to_cam.txt   <--- From the [calibration] download
    ├── calib_imu_to_velo.txt
    ├── calib_velo_to_cam.txt
    └── <date>_drive_<drive>_sync/  <--- From [synced+rectified data], e.g., 2011_09_26_drive_0001_sync
        ├── image_00/
        ├── image_01/
        ├── image_02/
        ├── image_03/
        ├── oxts/
        └── velodyne_points/
```

> [!IMPORTANT]
> Make sure the calibration `.txt` files are located directly under the `<date>/` folder, **not** inside the `<date>_drive_<drive>_sync/` directory.

---

## 🚀 How to Run the Converter

Run the script by providing the path to your `datasets` directory, the sequence date, and the drive index:

```bash
chmod +x kitti_to_rosbag.py

# Run with arguments
./kitti_to_rosbag.py --base_path /path/to/datasets --date 2011_09_26 --drive 0001 --output my_kitti_output.bag
```

### Command-Line Arguments
*   `--base_path` / `-b` (str): Directory containing the `<date>` subfolders (default: `/home/mahboob-alam/perception_ws/src/datasets`).
*   `--date` / `-d` (str, **required**): The date folder name (e.g., `2011_09_26`).
*   `--drive` / `-r` (str, **required**): The drive index (e.g., `0001`).
*   `--output` / `-o` (str): Custom name for the generated bag file. Defaults to `kitti_<date>_drive_<drive>.bag`.

---

## 📡 Generated ROS Topics

The converter packages the raw dataset files into the following ROS topics:

| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/kitti/cam0/image_raw` | `sensor_msgs/Image` | Greyscale camera (left) |
| `/kitti/cam0/camera_info` | `sensor_msgs/CameraInfo` | Calibration for cam0 |
| `/kitti/cam1/image_raw` | `sensor_msgs/Image` | Greyscale camera (right) |
| `/kitti/cam1/camera_info` | `sensor_msgs/CameraInfo` | Calibration for cam1 |
| `/kitti/cam2/image_raw` | `sensor_msgs/Image` | Left color camera |
| `/kitti/cam2/camera_info` | `sensor_msgs/CameraInfo` | Calibration for cam2 |
| `/kitti/cam3/image_raw` | `sensor_msgs/Image` | Right color camera |
| `/kitti/cam3/camera_info` | `sensor_msgs/CameraInfo` | Calibration for cam3 |
| `/kitti/velo/pointcloud` | `sensor_msgs/PointCloud2` | Velodyne 3D point cloud scans |
| `/kitti/oxts/imu` | `sensor_msgs/Imu` | IMU orientations, angular velocities, and accelerations |
| `/kitti/oxts/gps` | `sensor_msgs/NavSatFix` | GPS position coordinates |
| `/kitti/oxts/odom` | `nav_msgs/Odometry` | Global positioning/odometry pose |
| `/tf` | `tf2_msgs/TFMessage` | Static coordinate frame transformations |
