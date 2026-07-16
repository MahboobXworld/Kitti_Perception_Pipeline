import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

class OccupancyNetwork:
    def __init__(self, config):
        self.cfg = config or {}
        self.last_voxel_centroids = np.array([])
        self.last_voxel_colors = np.array([])

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

    def generate_occupancy_grid(self, raw_points, detected_objects, header):
        # Filter raw points to the configured bounds
        min_z = self.cfg.get("min_z", -2.0)
        max_z = self.cfg.get("max_z", 4.0)
        x_min_occ, x_max_occ = self.cfg.get("x_range", [-40.0, 40.0])
        y_min_occ, y_max_occ = self.cfg.get("y_range", [-40.0, 40.0])

        occ_mask = (
            (raw_points[:, 0] >= x_min_occ) & (raw_points[:, 0] <= x_max_occ) &
            (raw_points[:, 1] >= y_min_occ) & (raw_points[:, 1] <= y_max_occ) &
            (raw_points[:, 2] >= min_z) & (raw_points[:, 2] <= max_z)
        )
        occ_points = raw_points[occ_mask]

        if len(occ_points) == 0:
            self.last_voxel_centroids = np.array([])
            self.last_voxel_colors = np.array([])
            return MarkerArray()

        # Voxel downsampling
        voxel_size = self.cfg.get("voxel_size", 0.4)
        voxel_centroids = (np.floor(occ_points / voxel_size) + 0.5) * voxel_size
        _, unique_indices = np.unique(voxel_centroids, axis=0, return_index=True)
        voxel_centroids = voxel_centroids[unique_indices]

        # Voxel coloring
        bg_col = self.cfg.get("class_colors", {}).get("bg_voxel", [0.2, 0.2, 0.4, 0.4])
        voxel_colors = np.tile(bg_col, (len(voxel_centroids), 1))

        # Color the ground voxels separately
        ground_z_thresh = -1.4
        ground_col = [0.15, 0.15, 0.25, 0.35]  # Darker slate blue for road
        voxel_colors[voxel_centroids[:, 2] <= ground_z_thresh] = ground_col

        # Color voxels inside detections
        class_colors = self.cfg.get("class_colors", {})
        default_color = [1.0, 1.0, 0.0, 0.8]  # Yellow default

        for obj in detected_objects:
            cx, cy, cz = obj["centroid_velo"]
            l, w, h = self.get_class_dimensions(obj["class_name"])

            # Shift centroid away from ego vehicle by half length to match the volumetric center
            d_xy = np.sqrt(cx**2 + cy**2)
            if d_xy > 1.0:
                cx_adj = cx + (cx / d_xy) * (l / 2.0)
                cy_adj = cy + (cy / d_xy) * (l / 2.0)
            else:
                cx_adj = cx
                cy_adj = cy

            # Find voxels inside the adjusted 3D bounding box
            mask = (
                (voxel_centroids[:, 0] >= cx_adj - l/2) & (voxel_centroids[:, 0] <= cx_adj + l/2) &
                (voxel_centroids[:, 1] >= cy_adj - w/2) & (voxel_centroids[:, 1] <= cy_adj + w/2) &
                (voxel_centroids[:, 2] >= cz - h/2) & (voxel_centroids[:, 2] <= cz + h/2)
            )

            color = class_colors.get(obj["class_name"], default_color[:3])
            if len(color) == 3:
                color = color + [0.85]  # Add opacity
            voxel_colors[mask] = color

        # Construct Voxel Grid CUBE_LIST Marker
        voxel_marker = Marker()
        voxel_marker.header.frame_id = header.frame_id  # "velodyne"
        voxel_marker.header.stamp = header.stamp
        voxel_marker.ns = "occupancy_grid"
        voxel_marker.id = 0
        voxel_marker.type = Marker.CUBE_LIST
        voxel_marker.action = Marker.ADD
        voxel_marker.scale.x = voxel_size
        voxel_marker.scale.y = voxel_size
        voxel_marker.scale.z = voxel_size
        voxel_marker.pose.orientation.w = 1.0

        for pt, col in zip(voxel_centroids, voxel_colors):
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])
            voxel_marker.points.append(p)

            c = ColorRGBA()
            c.r = float(col[0])
            c.g = float(col[1])
            c.b = float(col[2])
            c.a = float(col[3])
            voxel_marker.colors.append(c)

        self.last_voxel_centroids = voxel_centroids
        self.last_voxel_colors = voxel_colors

        marker_array = MarkerArray()
        marker_array.markers.append(voxel_marker)
        return marker_array

    def generate_object_markers(self, detected_objects, header):
        class_colors = self.cfg.get("class_colors", {})
        default_color = [1.0, 1.0, 0.0, 0.8]  # Yellow default

        marker_array = MarkerArray()

        # Add Ego Vehicle Box (Red) at base_link
        ego_l, ego_w, ego_h = (4.2, 1.8, 1.5)
        
        # 3D Box Marker (CUBE)
        ego_box = Marker()
        ego_box.header.frame_id = "base_link"
        ego_box.header.stamp = header.stamp
        ego_box.ns = "ego_vehicle_box"
        ego_box.id = 0
        ego_box.type = Marker.CUBE
        ego_box.action = Marker.ADD
        ego_box.pose.position.x = 0.0
        ego_box.pose.position.y = 0.0
        ego_box.pose.position.z = ego_h / 2.0  # sits on the ground
        ego_box.pose.orientation.w = 1.0
        ego_box.scale.x = ego_l
        ego_box.scale.y = ego_w
        ego_box.scale.z = ego_h
        ego_box.color.r = 1.0
        ego_box.color.g = 0.0
        ego_box.color.b = 0.0
        ego_box.color.a = 0.4

        # 3D Box Outline (wireframe)
        ego_wireframe = Marker()
        ego_wireframe.header.frame_id = "base_link"
        ego_wireframe.header.stamp = header.stamp
        ego_wireframe.ns = "ego_vehicle_wireframe"
        ego_wireframe.id = 0
        ego_wireframe.type = Marker.LINE_LIST
        ego_wireframe.action = Marker.ADD
        ego_wireframe.scale.x = 0.04  # line thickness
        ego_wireframe.pose.orientation.w = 1.0
        ego_wireframe.color.r = 1.0
        ego_wireframe.color.g = 0.0
        ego_wireframe.color.b = 0.0
        ego_wireframe.color.a = 1.0

        dx, dy, dz = ego_l / 2, ego_w / 2, ego_h / 2
        cz = ego_h / 2.0
        ego_corners = [
            Point(-dx, -dy, cz - dz),
            Point(dx, -dy, cz - dz),
            Point(dx, dy, cz - dz),
            Point(-dx, dy, cz - dz),
            Point(-dx, -dy, cz + dz),
            Point(dx, -dy, cz + dz),
            Point(dx, dy, cz + dz),
            Point(-dx, dy, cz + dz),
        ]
        edges = [
            (0,1), (1,2), (2,3), (3,0), # bottom
            (4,5), (5,6), (6,7), (7,4), # top
            (0,4), (1,5), (2,6), (3,7)  # vertical pillars
        ]
        for edge in edges:
            ego_wireframe.points.append(ego_corners[edge[0]])
            ego_wireframe.points.append(ego_corners[edge[1]])

        # Ego Text Label
        ego_text = Marker()
        ego_text.header.frame_id = "base_link"
        ego_text.header.stamp = header.stamp
        ego_text.ns = "ego_vehicle_label"
        ego_text.id = 0
        ego_text.type = Marker.TEXT_VIEW_FACING
        ego_text.action = Marker.ADD
        ego_text.pose.position.x = 0.0
        ego_text.pose.position.y = 0.0
        ego_text.pose.position.z = ego_h + 0.6
        ego_text.pose.orientation.w = 1.0
        ego_text.scale.z = 0.8
        ego_text.color.r = 1.0
        ego_text.color.g = 1.0
        ego_text.color.b = 1.0
        ego_text.color.a = 1.0
        ego_text.text = "EGO VEHICLE"

        marker_array.markers.append(ego_box)
        marker_array.markers.append(ego_wireframe)
        marker_array.markers.append(ego_text)

        for i, obj in enumerate(detected_objects):
            cx, cy, cz = obj["centroid_velo"]
            l, w, h = self.get_class_dimensions(obj["class_name"])
            color = class_colors.get(obj["class_name"], default_color[:3])

            # Shift centroid away from ego vehicle by half length to match the volumetric center
            d_xy = np.sqrt(cx**2 + cy**2)
            if d_xy > 1.0:
                cx_adj = cx + (cx / d_xy) * (l / 2.0)
                cy_adj = cy + (cy / d_xy) * (l / 2.0)
            else:
                cx_adj = cx
                cy_adj = cy

            # 3D Box Marker (CUBE)
            box_marker = Marker()
            box_marker.header.frame_id = header.frame_id
            box_marker.header.stamp = header.stamp
            box_marker.ns = "object_3d_boxes"
            box_marker.id = obj["id"]
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = cx_adj
            box_marker.pose.position.y = cy_adj
            box_marker.pose.position.z = cz
            box_marker.pose.orientation.w = 1.0
            box_marker.scale.x = l
            box_marker.scale.y = w
            box_marker.scale.z = h
            
            box_marker.color.r = float(color[0])
            box_marker.color.g = float(color[1])
            box_marker.color.b = float(color[2])
            box_marker.color.a = 0.35  # Semi-transparent box

            # 3D Box Outline (using wireframe, type: LINE_LIST)
            wireframe = Marker()
            wireframe.header.frame_id = header.frame_id
            wireframe.header.stamp = header.stamp
            wireframe.ns = "object_3d_wireframes"
            wireframe.id = obj["id"]
            wireframe.type = Marker.LINE_LIST
            wireframe.action = Marker.ADD
            wireframe.scale.x = 0.04  # line thickness
            wireframe.pose.orientation.w = 1.0
            
            wireframe.color.r = float(color[0])
            wireframe.color.g = float(color[1])
            wireframe.color.b = float(color[2])
            wireframe.color.a = 1.0

            # Define 8 corners of the box centered at cx_adj, cy_adj, cz
            dx, dy, dz = l / 2, w / 2, h / 2
            corners = [
                Point(cx_adj - dx, cy_adj - dy, cz - dz),
                Point(cx_adj + dx, cy_adj - dy, cz - dz),
                Point(cx_adj + dx, cy_adj + dy, cz - dz),
                Point(cx_adj - dx, cy_adj + dy, cz - dz),
                Point(cx_adj - dx, cy_adj - dy, cz + dz),
                Point(cx_adj + dx, cy_adj - dy, cz + dz),
                Point(cx_adj + dx, cy_adj + dy, cz + dz),
                Point(cx_adj - dx, cy_adj + dy, cz + dz),
            ]
            # 12 edges
            edges = [
                (0,1), (1,2), (2,3), (3,0), # bottom
                (4,5), (5,6), (6,7), (7,4), # top
                (0,4), (1,5), (2,6), (3,7)  # vertical pillars
            ]
            for edge in edges:
                wireframe.points.append(corners[edge[0]])
                wireframe.points.append(corners[edge[1]])

            # Text Label Marker
            text_marker = Marker()
            text_marker.header.frame_id = header.frame_id
            text_marker.header.stamp = header.stamp
            text_marker.ns = "object_labels"
            text_marker.id = obj["id"]
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = cx_adj
            text_marker.pose.position.y = cy_adj
            text_marker.pose.position.z = cz + h/2 + 0.6  # Float above box
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.8  # Font size
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"{obj['class_name'].upper()} #{obj['id']}\nDist: {obj['depth']:.1f}m"

            marker_array.markers.append(box_marker)
            marker_array.markers.append(wireframe)
            marker_array.markers.append(text_marker)

        return marker_array
