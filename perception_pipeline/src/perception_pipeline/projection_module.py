from __future__ import annotations

from dataclasses import dataclass

import logging
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ProjectionError(Exception):
    pass


@dataclass
class ProjectionResult:
    """
    Container for projected point cloud outputs:
    - pixels: (N, 2) image coordinates (u, v)
    - depth: (N,) depth values in camera frame
    - indices: (N,) indices mapping back to the input cloud
    - cam_points: (N, 3) 3D coordinates in the camera coordinate frame
    """
    pixels: np.ndarray
    depth: np.ndarray
    indices: np.ndarray
    cam_points: np.ndarray


class ProjectionModule:
    """
    LiDAR to camera projection utility.
    Supports pinhole, fisheye, homogeneous transform matrices,
    rectification, distortion models, and occlusion filters.
    """

    def __init__(self, config: dict) -> None:
        # Load camera intrinsics
        fx, fy, cx, cy = config["intrinsics"]
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        self.image_width, self.image_height = config["resolution"]

        self.K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Camera lens model parameters
        self.camera_model = config.get("camera_model", "pinhole")
        self.distortion_model = config.get("distortion_model", "none")
        self.D = np.array(
            config.get("distortion_coeffs", []),
            dtype=np.float64,
        ).flatten()

        # Rig-to-camera extrinsic transform matrix
        T = np.array(
            config["extrinsic"]["transform_matrix"],
            dtype=np.float64,
        )

        if T.shape != (4, 4):
            raise ProjectionError("transform_matrix must be 4x4")

        if not np.allclose(T[3], [0, 0, 0, 1]):
            raise ProjectionError("last row must be [0,0,0,1]")

        self.T = T
        self.R = T[:3, :3]
        self.t = T[:3, 3]

        det = np.linalg.det(self.R)
        if not np.isclose(det, 1.0, atol=1e-3):
            raise ProjectionError(f"invalid rotation matrix determinant: {det}")

        # Optional rectification matrix (e.g. for stereo alignment in KITTI)
        self.R_rect = None
        if "rectification_matrix" in config:
            self.R_rect = np.array(
                config["rectification_matrix"],
                dtype=np.float64,
            )
            if self.R_rect.shape != (4, 4):
                raise ProjectionError("rectification_matrix must be 4x4")

        # Optional projection matrix (3x4)
        self.P = None
        if "projection_matrix" in config:
            self.P = np.array(
                config["projection_matrix"],
                dtype=np.float64,
            )
            if self.P.shape != (3, 4):
                raise ProjectionError("projection_matrix must be 3x4")

        # Visual distance clipping limits
        self.min_distance = float(config.get("min_distance", 0.5))
        self.max_distance = float(config.get("max_distance", 100.0))
        self.enable_occlusion = bool(config.get("enable_occlusion", True))

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectionModule":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg["projection"])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        # Transform points from LiDAR frame to camera frame
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        points_h = np.hstack((points, ones))
        cam_h = points_h @ self.T.T
        return cam_h[:, :3]

    def rectify_points(self, cam_points: np.ndarray) -> np.ndarray:
        if self.R_rect is None:
            return cam_points

        ones = np.ones((cam_points.shape[0], 1), dtype=np.float64)
        cam_h = np.hstack((cam_points, ones))
        rect_h = cam_h @ self.R_rect.T
        return rect_h[:, :3]

    def project_to_image(self, points: np.ndarray) -> ProjectionResult:
        # Validate point cloud format
        if not isinstance(points, np.ndarray):
            raise ProjectionError("points must be numpy array")

        if points.ndim != 2 or points.shape[1] < 3:
            raise ProjectionError("points must have shape (N,3+)")

        xyz = points[:, :3]
        xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        indices = np.arange(len(xyz), dtype=np.int32)

        # Drop NaN or Inf points
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        indices = indices[valid]

        if len(xyz) == 0:
            return self._empty()

        # Crop points outside range limits
        dist = np.linalg.norm(xyz, axis=1)
        in_range = (dist >= self.min_distance) & (dist <= self.max_distance)
        xyz = xyz[in_range]
        indices = indices[in_range]

        if len(xyz) == 0:
            return self._empty()

        # Transform to camera coordinate frame
        cam_points = self.transform_points(xyz)

        # Keep points in front of camera
        front = cam_points[:, 2] > 0.01
        cam_points = cam_points[front]
        indices = indices[front]

        if len(cam_points) == 0:
            return self._empty()

        # Apply rectification if defined
        cam_points = self.rectify_points(cam_points)
        depth = cam_points[:, 2].copy()

        # Project 3D coordinates onto 2D image coordinates
        if self.P is not None:
            pixels = self._project_matrix(cam_points)
        elif self.camera_model == "fisheye":
            pixels = self._fisheye(cam_points)
        else:
            pixels = self._pinhole(cam_points)

        # Filter out points projecting outside image bounds
        u = pixels[:, 0]
        v = pixels[:, 1]
        in_image = (
            (u >= 0)
            & (u < self.image_width)
            & (v >= 0)
            & (v < self.image_height)
        )

        pixels = pixels[in_image]
        depth = depth[in_image]
        cam_points = cam_points[in_image]
        indices = indices[in_image]

        if len(pixels) == 0:
            return self._empty()

        # Apply Z-buffer occlusion filter to hide points behind foreground objects
        if self.enable_occlusion:
            keep = self._occlusion_filter(pixels, depth)
            pixels = pixels[keep]
            depth = depth[keep]
            cam_points = cam_points[keep]
            indices = indices[keep]

        return ProjectionResult(
            pixels=pixels.astype(np.float32),
            depth=depth.astype(np.float32),
            indices=indices.astype(np.int32),
            cam_points=cam_points.astype(np.float32),
        )

    def _project_matrix(self, cam_points: np.ndarray) -> np.ndarray:
        # Project using a custom 3x4 projection matrix (e.g. KITTI calibration format)
        ones = np.ones((cam_points.shape[0], 1), dtype=np.float64)
        cam_h = np.hstack((cam_points, ones))
        img = cam_h @ self.P.T

        z = img[:, 2]
        valid = np.abs(z) > 1e-8

        u = np.full_like(z, np.nan)
        v = np.full_like(z, np.nan)

        u[valid] = img[valid, 0] / z[valid]
        v[valid] = img[valid, 1] / z[valid]

        return np.stack((u, v), axis=1)

    def _pinhole(self, cam_points: np.ndarray) -> np.ndarray:
        # Standard pinhole camera model projection
        X = cam_points[:, 0]
        Y = cam_points[:, 1]
        Z = cam_points[:, 2]

        x = X / Z
        y = Y / Z

        # Apply plumb_bob distortion correction if coefficients exist
        if (
            self.distortion_model == "plumb_bob"
            and len(self.D) >= 4
        ):
            k1, k2, p1, p2 = self.D[:4]
            r2 = x**2 + y**2
            radial = (1 + k1 * r2 + k2 * r2**2)
            x_dist = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2))
            y_dist = (y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y)
            x = x_dist
            y = y_dist

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        return np.stack((u, v), axis=1)

    def _fisheye(self, cam_points: np.ndarray) -> np.ndarray:
        # Standard fisheye distortion projection
        X = cam_points[:, 0]
        Y = cam_points[:, 1]
        Z = cam_points[:, 2]

        x = X / Z
        y = Y / Z

        r = np.sqrt(x**2 + y**2)
        r_safe = np.where(r < 1e-8, 1e-8, r)
        theta = np.arctan(r)

        if len(self.D) >= 4:
            k1, k2, k3, k4 = self.D[:4]
        else:
            k1 = k2 = k3 = k4 = 0.0

        theta2 = theta**2
        theta_d = theta * (
            1
            + k1 * theta2
            + k2 * theta2**2
            + k3 * theta2**3
            + k4 * theta2**4
        )

        scale = theta_d / r_safe
        u = self.fx * x * scale + self.cx
        v = self.fy * y * scale + self.cy

        return np.stack((u, v), axis=1)

    def _occlusion_filter(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        # Basic Z-buffer filtering: keeps the point with minimum depth per pixel
        u = pixels[:, 0].astype(np.int32)
        v = pixels[:, 1].astype(np.int32)

        linear = v * self.image_width + u
        order = np.argsort(depth)

        _, first = np.unique(
            linear[order],
            return_index=True,
        )

        keep_idx = order[first]
        mask = np.zeros(len(pixels), dtype=bool)
        mask[keep_idx] = True

        return mask

    @staticmethod
    def _empty() -> ProjectionResult:
        return ProjectionResult(
            pixels=np.empty((0, 2), dtype=np.float32),
            depth=np.empty((0,), dtype=np.float32),
            indices=np.empty((0,), dtype=np.int32),
            cam_points=np.empty((0, 3), dtype=np.float32),
        )