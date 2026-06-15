#!/usr/bin/env python3

import rospy
import yaml
import time
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2 as pc2

from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Point
from sensor_msgs.msg import RegionOfInterest

from perception_pipeline.msg import Objects, Object  

from perception_pipeline.lidar_filter import LidarFilter
from perception_pipeline.yolo_detector import YoloDetector
from perception_pipeline.projection_module import ProjectionModule


class SensorFusionNode:

    def __init__(self):

        # =====================================================
        # Load config
        # =====================================================
        config_path = rospy.get_param("/fusion_config_path")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        fusion_cfg = config["fusion"]

        rospy.loginfo("Fusion config loaded successfully")

        # =====================================================
        # Module configs
        # =====================================================
        lidar_path = rospy.get_param("/lidar_config_path")
        yolo_path = rospy.get_param("/yolo_config_path")
        proj_path = rospy.get_param("/projection_config_path")

        def load_yaml(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)

        lidar_cfg = load_yaml(lidar_path)["lidar_filter"]
        yolo_cfg = load_yaml(yolo_path)["yolo_detector"]
        proj_cfg = load_yaml(proj_path)["projection"]

        # =====================================================
        # Modules
        # =====================================================
        self.lidar_filter = LidarFilter(lidar_cfg)
        self.yolo = YoloDetector(yolo_cfg)
        self.projection = ProjectionModule(proj_cfg)

        self.bridge = CvBridge()

        # =====================================================
        # Topics
        # =====================================================
        lidar_topic = rospy.get_param("topics/lidar", fusion_cfg["subscribe"]["lidar"])
        image_topic = rospy.get_param("topics/image", fusion_cfg["subscribe"]["image"])

        fusion_image_topic = rospy.get_param("topics/output", fusion_cfg["publish"]["fusion_image"])
        objects_topic = rospy.get_param("topics/objects", fusion_cfg["publish"]["objects"])

        # =====================================================
        # Sync
        # =====================================================
        queue_size = fusion_cfg["sync"]["queue_size"]
        slop = fusion_cfg["sync"]["slop"]

        # =====================================================
        # Visualization
        # =====================================================
        self.visualization_enable = fusion_cfg["visualization"]["enable"]

        self.colormap_name = fusion_cfg["visualization"]["depth_color_map"]

        self.colormap_dict = {
            "HSV": cv2.COLORMAP_HSV,
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "COOL": cv2.COLORMAP_COOL,
            "OCEAN": cv2.COLORMAP_OCEAN,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS
        }

        self.depth_colormap = self.colormap_dict.get(
            self.colormap_name,
            cv2.COLORMAP_HSV
        )

        # =====================================================
        # Subscribers
        # =====================================================
        self.lidar_sub = Subscriber(lidar_topic, PointCloud2)
        self.image_sub = Subscriber(image_topic, Image)

        self.sync = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.image_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=False
        )

        self.sync.registerCallback(self.callback)

        # =====================================================
        # Publishers
        # =====================================================
        self.image_pub = rospy.Publisher(
            fusion_image_topic,
            Image,
            queue_size=1
        )

        self.objects_pub = rospy.Publisher(
            objects_topic,
            Objects,
            queue_size=1
        )

        rospy.loginfo("Sensor Fusion Node READY")

    # =====================================================
    # Callback
    # =====================================================
    def callback(self, lidar_msg, image_msg):

        t0 = time.time()

        try:

            # -------------------------------------------------
            # Image
            # -------------------------------------------------
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # -------------------------------------------------
            # LiDAR filter
            # -------------------------------------------------
            points = self.lidar_filter.filter(lidar_msg)

            if len(points) == 0:
                return

            # -------------------------------------------------
            # YOLO
            # -------------------------------------------------
            det_out = self.yolo.detect(image)

            # -------------------------------------------------
            # Projection
            # -------------------------------------------------
            proj = self.projection.project_to_image(points)

            vis = image.copy()

            # =================================================
            # OBJECT BUILDING (NEW PART)
            # =================================================
            objects_msg = Objects()
            objects_msg.header = image_msg.header

            h_img, w_img = image.shape[:2]

            # -------------------------------------------------
            # For each detection, associate LiDAR points
            # -------------------------------------------------
            for det in det_out.detections:

                x1, y1, x2, y2 = det.bbox.astype(int)

                # Clip bounding box to image boundaries to prevent negative values
                # or values exceeding image size, which would crash ROS serialization
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img - 1))
                y2 = max(0, min(y2, h_img - 1))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Vectorized point-in-box check
                mask = (
                    (proj.pixels[:, 0] >= x1) & (proj.pixels[:, 0] <= x2) &
                    (proj.pixels[:, 1] >= y1) & (proj.pixels[:, 1] <= y2)
                )

                inside_depths = proj.depth[mask]

                if len(inside_depths) == 0:
                    continue

                avg_depth = float(np.median(inside_depths))

                # Compute 3D centroid in camera frame
                inside_cam_points = proj.cam_points[mask]
                centroid_cam = np.mean(inside_cam_points, axis=0)

                obj = Object()
                obj.id = det.track_id 
                obj.class_name = det.class_name
                obj.confidence = float(det.confidence)
                obj.depth = avg_depth

                # physical 3D coordinates in camera frame (meters)
                obj.position = Point()
                obj.position.x = float(centroid_cam[0])
                obj.position.y = float(centroid_cam[1])
                obj.position.z = float(centroid_cam[2])

                obj.bbox = RegionOfInterest()
                obj.bbox.x_offset = int(x1)
                obj.bbox.y_offset = int(y1)
                obj.bbox.width = int(x2 - x1)
                obj.bbox.height = int(y2 - y1)

                objects_msg.objects.append(obj)

            # -------------------------------------------------
            # Publish objects
            # -------------------------------------------------
            self.objects_pub.publish(objects_msg)

            # =================================================
            # Visualization
            # =================================================
            if self.visualization_enable and len(proj.depth) > 0:

                # Vectorized color mapping
                depths_clipped = np.clip(proj.depth, 1, 30)
                norm_depths = (depths_clipped / 30.0 * 255).astype(np.uint8)
                
                colors = cv2.applyColorMap(
                    norm_depths[:, None],
                    self.depth_colormap
                )
                colors = colors.squeeze(axis=1)

                for (u, v), color in zip(proj.pixels, colors):
                    cv2.circle(vis, (int(u), int(v)), 2, tuple(int(x) for x in color), -1)

                for det in det_out.detections:

                    x1, y1, x2, y2 = det.bbox.astype(int)
                    # Clip for drawing as well
                    x1 = max(0, min(x1, w_img - 1))
                    y1 = max(0, min(y1, h_img - 1))
                    x2 = max(0, min(x2, w_img - 1))
                    y2 = max(0, min(y2, h_img - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"{det.class_name} {det.confidence:.2f}"

                    # Show tracker ID only if tracking is enabled
                    if self.yolo.enable_tracking and det.track_id >= 0:
                        label += f" ID:{det.track_id}"

                    cv2.putText(
                        vis,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # -------------------------------------------------
            # Publish fused image
            # -------------------------------------------------
            out = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            out.header = image_msg.header
            self.image_pub.publish(out)

            dt = (time.time() - t0) * 1000

            rospy.loginfo_throttle(
                1,
                f"Fusion: {dt:.1f} ms | "
                f"Points: {len(points)} | "
                f"Detections: {len(det_out.detections)} | "
                f"Objects: {len(objects_msg.objects)}"
            )

        except Exception as e:
            rospy.logerr(f"Fusion error: {e}")

    # =====================================================
    # Depth color
    # =====================================================
    def depth_color(self, d):

        d = np.clip(d, 1, 30)
        norm = d / 30.0

        color = cv2.applyColorMap(
            np.uint8([[norm * 255]]),
            self.depth_colormap
        )

        return tuple(int(x) for x in color[0][0])


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    rospy.init_node("fusion_node")

    SensorFusionNode()

    rospy.spin()