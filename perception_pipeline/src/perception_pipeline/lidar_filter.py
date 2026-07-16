import numpy as np
import sensor_msgs.point_cloud2 as pc2


class LidarFilter:
    def __init__(self, config):
        self.cfg = config or {}

    def cloud_to_numpy(self, cloud_msg):
        # Read PointCloud2 coordinates into a numpy array
        points = np.array(
            list(pc2.read_points(
                cloud_msg,
                field_names=("x", "y", "z"),
                skip_nans=False
            )),
            dtype=np.float32
        )

        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        return points

    def remove_invalid(self, points):
        # Clean up any NaN/Inf coordinates
        if not (self.cfg.get("remove_nan", True) or self.cfg.get("remove_inf", True)):
            return points

        return points[np.isfinite(points).all(axis=1)]

    def radial_filter(self, points):
        # Keep points within minimum/maximum range limits
        min_r = self.cfg.get("min_range", 0.0)
        max_r = self.cfg.get("max_range", -1.0)

        if max_r <= 0:
            return points

        dist_sq = np.sum(points * points, axis=1)

        return points[
            (dist_sq >= min_r * min_r) &
            (dist_sq <= max_r * max_r)
        ]

    def roi_filter(self, points):
        # Crop to the 3D bounding box / region of interest
        x_min = self.cfg.get("x_min", -50.0)
        x_max = self.cfg.get("x_max", 50.0)

        y_min = self.cfg.get("y_min", -50.0)
        y_max = self.cfg.get("y_max", 50.0)

        z_min = self.cfg.get("z_min", -5.0)
        z_max = self.cfg.get("z_max", 5.0)

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )

        return points[mask]

    def ground_filter(self, points):
        # Filter out points below the road floor threshold
        if not self.cfg.get("ground_filter", False):
            return points

        z_thresh = self.cfg.get("ground_z", -1.5)
        return points[points[:, 2] > z_thresh]

    def random_downsample(self, points):
        # Downsample points by randomly selecting a subset fraction
        ds_cfg = self.cfg.get("downsample", {})
        percentage = ds_cfg.get("percentage", 1.0)

        if percentage >= 1.0 or len(points) == 0:
            return points

        seed = self.cfg.get("random_seed", 42)
        np.random.seed(seed)

        n = int(len(points) * percentage)
        if n <= 0:
            return points

        idx = np.random.choice(len(points), n, replace=False)
        return points[idx]

    def voxel_downsample(self, points):
        # Downsample by grouping points in 3D voxels and taking their centroids
        ds_cfg = self.cfg.get("downsample", {})
        leaf = ds_cfg.get("leaf_size", 0.2)

        if len(points) == 0:
            return points

        voxel = np.floor(points / leaf)

        _, inverse, counts = np.unique(
            voxel,
            axis=0,
            return_inverse=True,
            return_counts=True
        )

        voxel_sum = np.zeros((counts.shape[0], 3), dtype=np.float32)
        np.add.at(voxel_sum, inverse, points)

        return voxel_sum / counts[:, None]

    def downsample(self, points):
        # Choose downsampling method based on config
        ds_cfg = self.cfg.get("downsample", {})

        if not ds_cfg.get("enable", False):
            return points

        method = ds_cfg.get("method", "random")

        if method == "random":
            return self.random_downsample(points)

        elif method == "voxel":
            return self.voxel_downsample(points)

        return points

    def filter(self, cloud_msg):
        # Process the point cloud through the filtering stages
        points = self.cloud_to_numpy(cloud_msg)

        if len(points) == 0:
            return points

        points = self.remove_invalid(points)
        points = self.radial_filter(points)
        points = self.roi_filter(points)
        points = self.ground_filter(points)
        points = self.downsample(points)

        return points