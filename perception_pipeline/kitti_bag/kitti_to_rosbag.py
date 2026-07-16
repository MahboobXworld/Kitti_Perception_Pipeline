#!/usr/bin/env python3

import os
import cv2
import argparse
import rospy
import rosbag
import pykitti
import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import (
    Image,
    Imu,
    NavSatFix,
    CameraInfo,
    PointCloud2,
    PointField
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header

import sensor_msgs.point_cloud2 as pcl2
import tf.transformations as tf_trans

bridge = CvBridge()


def create_camera_info(K, width, height, frame_id, stamp):
    cam_info = CameraInfo()
    cam_info.header.stamp = stamp
    cam_info.header.frame_id = frame_id

    cam_info.width = width
    cam_info.height = height

    cam_info.K = K.flatten().tolist()
    cam_info.P = [
        K[0,0], 0, K[0,2], 0,
        0, K[1,1], K[1,2], 0,
        0, 0, 1, 0
    ]

    cam_info.R = np.eye(3).flatten().tolist()
    cam_info.distortion_model = "plumb_bob"
    cam_info.D = [0,0,0,0,0]

    return cam_info


def create_pointcloud(points, stamp, frame_id):
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
    ]

    return pcl2.create_cloud(header, fields, points)


def publish_tf(bag, stamp):
    tf_msg = TFMessage()
    transforms = []

    # base_link -> imu
    t1 = TransformStamped()
    t1.header.stamp = stamp
    t1.header.frame_id = "base_link"
    t1.child_frame_id = "imu_link"
    t1.transform.rotation.w = 1.0
    transforms.append(t1)

    # base_link -> velodyne
    t2 = TransformStamped()
    t2.header.stamp = stamp
    t2.header.frame_id = "base_link"
    t2.child_frame_id = "velodyne"
    t2.transform.translation.x = 0.0
    t2.transform.translation.y = 0.0
    t2.transform.translation.z = 1.73
    t2.transform.rotation.w = 1.0
    transforms.append(t2)

    # base_link -> cameras
    for cam in range(4):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "base_link"
        t.child_frame_id = f"cam{cam}"
        t.transform.rotation.w = 1.0
        transforms.append(t)

    tf_msg.transforms = transforms
    bag.write('/tf', tf_msg, stamp)


def main():
    parser = argparse.ArgumentParser(description="Convert raw KITTI dataset to ROS Bag format.")
    parser.add_argument(
        "--base_path", "-b",
        type=str,
        default="/home/mahboob-alam/perception_ws/src/datasets",
        help="Path to the directory containing the dataset folders."
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="Date of the sequence (e.g. 2011_10_03)."
    )
    parser.add_argument(
        "--drive", "-r",
        type=str,
        required=True,
        help="Drive index of the sequence (e.g. 0047)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output bag file path. Defaults to 'kitti_<date>_drive_<drive>.bag'."
    )
    args = parser.parse_args()

    # Determine output bag path
    if args.output:
        output_bag = args.output
    else:
        output_bag = f"kitti_{args.date}_drive_{args.drive}.bag"

    print(f"Loading KITTI dataset from: {args.base_path}")
    print(f"Sequence: Date {args.date}, Drive {args.drive}")

    dataset = pykitti.raw(args.base_path, args.date, args.drive)
    bag = rosbag.Bag(output_bag, 'w')

    K_cam2 = dataset.calib.K_cam2
    K_cam3 = dataset.calib.K_cam3

    try:
        for i in range(len(dataset.timestamps)):
            stamp = rospy.Time.from_sec(dataset.timestamps[i].timestamp())
            print(f"Processing frame {i}/{len(dataset.timestamps)-1}")

            # Static frame transformations
            publish_tf(bag, stamp)

            # Cameras (cam0 to cam3)
            cams = [
                ("cam0", dataset.get_cam0(i), "mono8"),
                ("cam1", dataset.get_cam1(i), "mono8"),
                ("cam2", dataset.get_cam2(i), "bgr8"),
                ("cam3", dataset.get_cam3(i), "bgr8")
            ]

            for cam_name, img, enc in cams:
                img_np = np.array(img)
                if enc != "mono8":
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                msg = bridge.cv2_to_imgmsg(img_np, encoding=enc)
                msg.header.stamp = stamp
                msg.header.frame_id = cam_name

                topic = f"/kitti/{cam_name}/image_raw"
                bag.write(topic, msg, stamp)

                # Camera calibration matrices Info
                h, w = img_np.shape[:2]
                if cam_name in ["cam2", "cam3"]:
                    K = K_cam2 if cam_name == "cam2" else K_cam3
                else:
                    K = np.eye(3)

                cam_info = create_camera_info(K, w, h, cam_name, stamp)
                bag.write(f"/kitti/{cam_name}/camera_info", cam_info, stamp)

            # Velodyne LiDAR pointcloud
            velo = dataset.get_velo(i)
            pcl_msg = create_pointcloud(velo, stamp, "velodyne")
            bag.write("/kitti/velo/pointcloud", pcl_msg, stamp)

            # OXTS GPS & IMU sensors
            oxts = dataset.oxts[i]
            packet = oxts.packet

            # IMU orientations and rates
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = "imu_link"

            q = tf_trans.quaternion_from_euler(
                packet.roll,
                packet.pitch,
                packet.yaw
            )

            imu_msg.orientation.x = q[0]
            imu_msg.orientation.y = q[1]
            imu_msg.orientation.z = q[2]
            imu_msg.orientation.w = q[3]

            imu_msg.angular_velocity.x = packet.wx
            imu_msg.angular_velocity.y = packet.wy
            imu_msg.angular_velocity.z = packet.wz

            imu_msg.linear_acceleration.x = packet.ax
            imu_msg.linear_acceleration.y = packet.ay
            imu_msg.linear_acceleration.z = packet.az

            bag.write("/kitti/oxts/imu", imu_msg, stamp)

            # GPS positioning coordinates
            gps_msg = NavSatFix()
            gps_msg.header.stamp = stamp
            gps_msg.header.frame_id = "gps"

            gps_msg.latitude = packet.lat
            gps_msg.longitude = packet.lon
            gps_msg.altitude = packet.alt

            bag.write("/kitti/oxts/gps", gps_msg, stamp)

            # Ground-truth global odometry
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "world"
            odom.child_frame_id = "base_link"

            odom.pose.pose.position.x = oxts.T_w_imu[0,3]
            odom.pose.pose.position.y = oxts.T_w_imu[1,3]
            odom.pose.pose.position.z = oxts.T_w_imu[2,3]

            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]

            bag.write("/kitti/oxts/odom", odom, stamp)

    finally:
        bag.close()
        print("\nDONE.")
        print(f"Saved bag file to: {output_bag}")


if __name__ == "__main__":
    # Ensure name node registration works if run inside a ROS launch context
    rospy.init_node("kitti_to_rosbag", anonymous=True)
    main()