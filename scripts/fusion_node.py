#!/usr/bin/env python3

import rospy
import yaml
import time
import numpy as np
import cv2
import tf
import tf.transformations

from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import RegionOfInterest

from perception_pipeline.msg import Objects, Object  

from perception_pipeline.lidar_filter import LidarFilter
from perception_pipeline.yolo_detector import YoloDetector
from perception_pipeline.projection_module import ProjectionModule
from perception_pipeline.occupancy_network import OccupancyNetwork


class SensorFusionNode:

    def __init__(self):
        # Load main node configurations
        config_path = rospy.get_param("/fusion_config_path")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        fusion_cfg = config["fusion"]
        rospy.loginfo("Fusion config loaded successfully")

        # Load configurations for the child modules
        lidar_path = rospy.get_param("/lidar_config_path")
        yolo_path = rospy.get_param("/yolo_config_path")
        proj_path = rospy.get_param("/projection_config_path")

        def load_yaml(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)

        lidar_cfg = load_yaml(lidar_path)["lidar_filter"]
        yolo_cfg = load_yaml(yolo_path)["yolo_detector"]
        proj_cfg = load_yaml(proj_path)["projection"]

        self.bev_cfg = fusion_cfg.get("bev", {})
        self.occupancy_cfg = fusion_cfg.get("occupancy_3d", {})

        # Initialize submodules
        self.lidar_filter = LidarFilter(lidar_cfg)
        self.yolo = YoloDetector(yolo_cfg)
        self.projection = ProjectionModule(proj_cfg)
        self.occupancy_net = OccupancyNetwork(self.occupancy_cfg)

        self.bridge = CvBridge()

        # Set up topic names
        lidar_topic = rospy.get_param("topics/lidar", fusion_cfg["subscribe"]["lidar"])
        image_topic = rospy.get_param("topics/image", fusion_cfg["subscribe"]["image"])

        fusion_image_topic = rospy.get_param("topics/output", fusion_cfg["publish"]["fusion_image"])
        objects_topic = rospy.get_param("topics/objects", fusion_cfg["publish"]["objects"])

        # Synchronization parameters
        queue_size = fusion_cfg["sync"]["queue_size"]
        slop = fusion_cfg["sync"]["slop"]

        # Colormap mapping for visualization
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

        # ----------------------------------------------------
        # NEW: BEV & Occupancy configurations
        # ----------------------------------------------------
        modules_cfg = fusion_cfg.get("modules", {})
        self.enable_2d_bev = modules_cfg.get("enable_2d_bev", True)
        self.enable_occupancy_net = modules_cfg.get("enable_occupancy_net", True)
        self.enable_3d_detections = modules_cfg.get("enable_3d_detections", True)

        # Trajectory history in world frame
        self.trajectory = []
        self.latest_odom = None

        # TF Broadcaster and Listener
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Cached velodyne-to-base_link transform to avoid tf lockups
        self.trans_velo_base = None
        self.rot_velo_base = None

        # ----------------------------------------------------
        # Publishers & Subscribers
        # ----------------------------------------------------
        # Odom subscriber to track vehicle position in world frame
        self.odom_sub = rospy.Subscriber(
            "/kitti/oxts/odom",
            Odometry,
            self.odom_callback
        )

        # Message synchronizer for camera & lidar
        self.lidar_sub = Subscriber(lidar_topic, PointCloud2)
        self.image_sub = Subscriber(image_topic, Image)

        self.sync = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.image_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=False
        )
        self.sync.registerCallback(self.callback)

        # Output publishers
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

        if self.enable_2d_bev:
            bev_topic = self.bev_cfg.get("publish_topic", "/fusion/bev_image")
            self.bev_pub = rospy.Publisher(
                bev_topic,
                Image,
                queue_size=1
            )

        if self.enable_occupancy_net:
            self.occupancy_pub = rospy.Publisher(
                "/fusion/occupancy_grid",
                MarkerArray,
                queue_size=1
            )

        if self.enable_3d_detections:
            self.detected_objects_3d_pub = rospy.Publisher(
                "/fusion/detected_objects_3d",
                MarkerArray,
                queue_size=1
            )

        # Register shutdown cleanup
        rospy.on_shutdown(self.cleanup)

        rospy.loginfo("Sensor Fusion Node READY with BEV and Occupancy Net support")

    def odom_callback(self, msg):
        self.latest_odom = msg
        # Broadcast the world -> base_link TF transform dynamically
        try:
            self.tf_broadcaster.sendTransform(
                (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
                (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
                msg.header.stamp,
                msg.child_frame_id,  # base_link
                msg.header.frame_id  # world
            )
        except Exception as e:
            rospy.logwarn_throttle(5, f"TF broadcast failed: {e}")

        # Update trajectory history
        self.trajectory.append((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
        max_len = self.bev_cfg.get("trajectory_length", 200)
        if len(self.trajectory) > max_len:
            self.trajectory.pop(0)

    def get_class_dimensions(self, class_name):
        if class_name == "car":
            return (4.2, 1.8, 1.5)  # length (x), width (y), height (z) in meters
        elif class_name in ["truck", "bus"]:
            return (8.0, 2.5, 3.0)
        elif class_name in ["pedestrian", "person"]:
            return (0.8, 0.8, 1.7)
        elif class_name in ["cyclist", "bicycle", "motorcycle"]:
            return (2.0, 0.8, 1.5)
        else:
            return (2.0, 1.5, 1.5)

    def get_velo_to_base_transform(self):
        if self.trans_velo_base is not None:
            return self.trans_velo_base, self.rot_velo_base

        try:
            self.tf_listener.waitForTransform("base_link", "velodyne", rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform("base_link", "velodyne", rospy.Time(0))
            self.trans_velo_base = np.array(trans)
            # convert quaternion to 3x3 rotation matrix
            self.rot_velo_base = tf.transformations.quaternion_matrix(rot)[:3, :3]
            return self.trans_velo_base, self.rot_velo_base
        except Exception as e:
            # Fallback if lookup fails: standard KITTI Velodyne position
            # Velodyne is at x=0, y=0, z=0.8 relative to base_link (approx)
            rospy.logwarn_throttle(5, f"Velo to base TF lookup failed, using fallback: {e}")
            return np.array([0.0, 0.0, -0.8]), np.eye(3)

    def generate_bev_image(self, voxel_centroids, voxel_colors, detected_objects):
        # Displays the 2D Top View by projecting the 3D voxel grid and detections
        w, h = self.bev_cfg.get("resolution", [800, 800])
        scale = self.bev_cfg.get("scale", 10.0)
        ego_x = self.bev_cfg.get("ego_x", w // 2)
        ego_y = self.bev_cfg.get("ego_y", h - 200)

        # Initialize canvas
        bev_img = np.zeros((h, w, 3), dtype=np.uint8)
        bev_img[:] = [15, 15, 18]  # Dark futuristic background

        # Draw voxel grid from 3D BEV (occupancy grid)
        if voxel_centroids is not None and len(voxel_centroids) > 0:
            voxel_size = self.occupancy_cfg.get("voxel_size", 0.4)
            for pt, col in zip(voxel_centroids, voxel_colors):
                u = int(ego_x - pt[1] * scale)
                v = int(ego_y - pt[0] * scale)

                sz = int(voxel_size * scale)
                sz = max(1, sz)

                # Convert RGB/RGBA to BGR
                bgr = [int(col[2] * 255), int(col[1] * 255), int(col[0] * 255)]
                
                # Check borders to avoid OOB draw errors
                if 0 <= u < w and 0 <= v < h:
                    cv2.rectangle(
                        bev_img,
                        (u - sz // 2, v - sz // 2),
                        (u + sz // 2, v + sz // 2),
                        bgr,
                        -1
                    )

        # Draw range rings
        for r in [10, 20, 30, 40, 50]:
            radius_px = int(r * scale)
            cv2.circle(bev_img, (ego_x, ego_y), radius_px, (40, 40, 45), 1)
            cv2.putText(
                bev_img,
                f"{r}m",
                (ego_x + radius_px + 2, ego_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (70, 70, 75),
                1,
                cv2.LINE_AA
            )

        # Draw axes
        cv2.line(bev_img, (ego_x, 0), (ego_x, h), (30, 30, 35), 1)
        cv2.line(bev_img, (0, ego_y), (w, ego_y), (30, 30, 35), 1)

        # Draw ego vehicle trajectory (in local base_link/velo frame)
        if len(self.trajectory) > 1 and self.latest_odom is not None:
            t_curr = np.array([
                self.latest_odom.pose.pose.position.x,
                self.latest_odom.pose.pose.position.y,
                self.latest_odom.pose.pose.position.z
            ])
            q = self.latest_odom.pose.pose.orientation
            R_curr = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

            traj_world = np.array(self.trajectory)
            # Transform to base_link
            traj_local = (traj_world - t_curr) @ R_curr

            # Convert to pixels
            u_traj = ego_x - (traj_local[:, 1] * scale)
            v_traj = ego_y - (traj_local[:, 0] * scale)

            # Assemble path points
            path_pts = np.stack((u_traj, v_traj), axis=1).astype(np.int32)
            # Filter within borders
            valid_idx = (path_pts[:, 0] >= 0) & (path_pts[:, 0] < w) & (path_pts[:, 1] >= 0) & (path_pts[:, 1] < h)
            if np.any(valid_idx):
                cv2.polylines(bev_img, [path_pts], False, (0, 140, 255), 2, cv2.LINE_AA)

        # Draw ego vehicle shape - RED Color
        veh_w = int(1.8 * scale)
        veh_h = int(4.2 * scale)
        x1_ego = ego_x - veh_w // 2
        y1_ego = ego_y - veh_h // 2
        cv2.rectangle(bev_img, (x1_ego, y1_ego), (x1_ego + veh_w, y1_ego + veh_h), (0, 0, 150), -1)
        cv2.rectangle(bev_img, (x1_ego, y1_ego), (x1_ego + veh_w, y1_ego + veh_h), (0, 0, 255), 2)
        # Heading arrow
        cv2.arrowedLine(bev_img, (ego_x, ego_y + 10), (ego_x, ego_y - 20), (255, 255, 255), 2, tipLength=0.3)

        # Draw detected objects (all green with high accuracy)
        color = (0, 255, 0)

        for obj in detected_objects:
            cx, cy, cz = obj["centroid_velo"]
            l, w_dim, h_dim = self.get_class_dimensions(obj["class_name"])

            # Shift centroid away from ego vehicle by half length to match the volumetric center
            d_xy = np.sqrt(cx**2 + cy**2)
            if d_xy > 1.0:
                cx_adj = cx + (cx / d_xy) * (l / 2.0)
                cy_adj = cy + (cy / d_xy) * (l / 2.0)
            else:
                cx_adj = cx
                cy_adj = cy

            # Map center to pixels
            obj_u = ego_x - int(cy_adj * scale)
            obj_v = ego_y - int(cx_adj * scale)

            box_w_px = int(w_dim * scale)
            box_h_px = int(l * scale)

            x1_px = obj_u - box_w_px // 2
            y1_px = obj_v - box_h_px // 2
            x2_px = obj_u + box_w_px // 2
            y2_px = obj_v + box_h_px // 2

            # Draw box
            cv2.rectangle(bev_img, (x1_px, y1_px), (x2_px, y2_px), color, 2)
            # Semi-transparent filling
            overlay = bev_img.copy()
            cv2.rectangle(overlay, (x1_px, y1_px), (x2_px, y2_px), color, -1)
            cv2.addWeighted(overlay, 0.15, bev_img, 0.85, 0, bev_img)

            # Draw texts
            label = f"{obj['class_name'].upper()} #{obj['id']}"
            dist_lbl = f"{obj['depth']:.1f}m"

            cv2.putText(
                bev_img,
                label,
                (x1_px, y1_px - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                bev_img,
                dist_lbl,
                (x1_px, y1_px - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        return bev_img



    def callback(self, lidar_msg, image_msg):
        t0 = time.time()

        try:
            # Convert ROS Image to OpenCV BGR representation
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Apply filters to point cloud
            points = self.lidar_filter.filter(lidar_msg)
            if len(points) == 0:
                return

            # Detect 2D objects
            det_out = self.yolo.detect(image)

            # Project remaining 3D points to the image plane
            proj = self.projection.project_to_image(points)

            vis = image.copy()

            # Assemble custom ROS message outputs
            objects_msg = Objects()
            objects_msg.header = image_msg.header

            h_img, w_img = image.shape[:2]

            detected_objects = []

            # Match projected point coordinates to 2D bounding boxes
            for det in det_out.detections:
                x1, y1, x2, y2 = det.bbox.astype(int)

                # Clip coords to boundaries to prevent negative values or out of frame crashes
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img - 1))
                y2 = max(0, min(y2, h_img - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Get mask of projected points inside the current bounding box
                mask = (
                    (proj.pixels[:, 0] >= x1) & (proj.pixels[:, 0] <= x2) &
                    (proj.pixels[:, 1] >= y1) & (proj.pixels[:, 1] <= y2)
                )

                inside_depths = proj.depth[mask]
                if len(inside_depths) == 0:
                    continue

                # Estimate distance using the median depth value
                avg_depth = float(np.median(inside_depths))

                # Calculate object 3D centroid in camera/velodyne frames with filtering
                inside_cam_points = proj.cam_points[mask]
                inside_lidar_points = points[proj.indices][mask]

                # Filter: Keep points within 1.5 meters of median depth to eliminate foreground/background bleed
                depth_mask = np.abs(proj.depth[mask] - avg_depth) < 1.5

                # Filter: Keep points above -1.4 meters in Z (velodyne frame) to eliminate road surface points
                z_mask = inside_lidar_points[:, 2] > -1.4

                combined_mask = depth_mask & z_mask
                if not np.any(combined_mask):
                    combined_mask = depth_mask

                if np.any(combined_mask):
                    inside_cam_points = inside_cam_points[combined_mask]
                    inside_lidar_points = inside_lidar_points[combined_mask]

                centroid_cam = np.mean(inside_cam_points, axis=0)
                centroid_velo = np.mean(inside_lidar_points[:, :3], axis=0)

                obj = Object()
                obj.id = det.track_id 
                obj.class_name = det.class_name
                obj.confidence = float(det.confidence)
                obj.depth = avg_depth

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

                detected_objects.append({
                    "id": det.track_id if det.track_id >= 0 else len(detected_objects) + 1,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "depth": avg_depth,
                    "centroid_velo": centroid_velo
                })

            # Publish the fused objects list
            self.objects_pub.publish(objects_msg)

            # Draw projections and boxes on visual frame
            if self.visualization_enable and len(proj.depth) > 0:
                # Colorize projected depth points (clipped from 1m to 30m)
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
                    x1 = max(0, min(x1, w_img - 1))
                    y1 = max(0, min(y1, h_img - 1))
                    x2 = max(0, min(x2, w_img - 1))
                    y2 = max(0, min(y2, h_img - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.class_name} {det.confidence:.2f}"

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

            # Publish annotated visual frame
            out = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            out.header = image_msg.header
            self.image_pub.publish(out)

            # ----------------------------------------------------
            # 3D Occupancy and 2D BEV calculation
            # ----------------------------------------------------
            voxel_centroids = None
            voxel_colors = None

            if self.enable_occupancy_net or self.enable_2d_bev or self.enable_3d_detections:
                raw_points = self.lidar_filter.cloud_to_numpy(lidar_msg)
                raw_points = self.lidar_filter.remove_invalid(raw_points)
            else:
                raw_points = None

            if (self.enable_occupancy_net or self.enable_2d_bev) and raw_points is not None:
                occ_grid = self.occupancy_net.generate_occupancy_grid(raw_points, detected_objects, lidar_msg.header)
                voxel_centroids = self.occupancy_net.last_voxel_centroids
                voxel_colors = self.occupancy_net.last_voxel_colors
                if self.enable_occupancy_net:
                    self.occupancy_pub.publish(occ_grid)

            if self.enable_3d_detections:
                obj_markers = self.occupancy_net.generate_object_markers(detected_objects, lidar_msg.header)
                self.detected_objects_3d_pub.publish(obj_markers)

            # ----------------------------------------------------
            # 2D BEV Top View Visualization
            # ----------------------------------------------------
            if self.enable_2d_bev:
                bev_img = self.generate_bev_image(voxel_centroids, voxel_colors, detected_objects)
                
                # Publish BEV Image
                bev_msg = self.bridge.cv2_to_imgmsg(bev_img, "bgr8")
                bev_msg.header = lidar_msg.header
                self.bev_pub.publish(bev_msg)

                # Show local popup window if configured
                if self.bev_cfg.get("show_window", False):
                    cv2.imshow("BEV 2D Top View", bev_img)
                    cv2.waitKey(1)

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

    def depth_color(self, d):
        d = np.clip(d, 1, 30)
        norm = d / 30.0

        color = cv2.applyColorMap(
            np.uint8([[norm * 255]]),
            self.depth_colormap
        )

        return tuple(int(x) for x in color[0][0])

    def cleanup(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("fusion_node")
    SensorFusionNode()
    rospy.spin()