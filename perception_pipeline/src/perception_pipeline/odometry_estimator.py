import numpy as np
import open3d as o3d
import rospy
import tf.transformations
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

class LidarImuOdometry:
    def __init__(self, config):
        self.cfg = config or {}
        self.odom_cfg = self.cfg.get("odometry", {})
        
        # ICP Parameters
        self.voxel_size = self.odom_cfg.get("icp_voxel_size", 0.5)
        self.max_distance = self.odom_cfg.get("icp_max_distance", 1.0)
        self.max_iterations = self.odom_cfg.get("icp_max_iterations", 30)
        
        # State: global pose of base_link in world frame (4x4 homogeneous matrix)
        self.T_world_base = np.eye(4)
        
        # Cache of previous LiDAR scan (Open3D PointCloud)
        self.prev_pcd = None
        
        # IMU state variables for integration
        self.last_imu_time = None
        # Accumulated relative position, velocity, and orientation since last LiDAR frame
        self.imu_rel_t = np.zeros(3)
        self.imu_rel_v = np.zeros(3)
        self.imu_rel_q = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        
        # Extrinsic transform T_base_velo (from base_link to velodyne)
        self.T_base_velo = None

    def set_base_to_velo_transform(self, trans, rot):
        """
        Sets the static transform from base_link to velodyne coordinate frame.
        """
        T = tf.transformations.quaternion_matrix(rot)
        T[:3, 3] = trans
        self.T_base_velo = T

    def process_imu(self, msg):
        """
        Integrates IMU measurements to maintain relative motion between LiDAR scans.
        """
        curr_time = msg.header.stamp
        if self.last_imu_time is None:
            self.last_imu_time = curr_time
            return
            
        dt = (curr_time - self.last_imu_time).to_sec()
        if dt <= 0:
            return
            
        self.last_imu_time = curr_time
        
        # 1. Integrate angular velocity in 3D using small-angle quaternion update
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        angle = np.linalg.norm(w) * dt
        if angle > 1e-6:
            axis = w / np.linalg.norm(w)
            dq = tf.transformations.quaternion_about_axis(angle, axis)
            self.imu_rel_q = tf.transformations.quaternion_multiply(self.imu_rel_q, dq)
            self.imu_rel_q /= np.linalg.norm(self.imu_rel_q)
            
        # 2. Integrate linear acceleration in 3D
        R = tf.transformations.quaternion_matrix(self.imu_rel_q)[:3, :3]
        a_local = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        
        # Correct for gravity (standard 9.81 m/s^2 along Z axis in local/relative coordinate system)
        a_rel = R @ a_local
        a_rel[2] -= 9.81
        
        # Double integrate relative translation
        self.imu_rel_t += self.imu_rel_v * dt + 0.5 * a_rel * (dt ** 2)
        self.imu_rel_v += a_rel * dt

    def process_lidar(self, points, header):
        """
        Aligns the current point cloud with the previous point cloud using Open3D ICP.
        Uses IMU prediction as the initial guess.
        
        points: N x 3 numpy array of points in velodyne (LiDAR) frame.
        returns: nav_msgs/Odometry message representing the current base_link pose in world frame.
        """
        # If no points, return the current state
        if len(points) == 0:
            return self.get_odom_msg(header)

        # 1. Convert points to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # Make sure we have the base_link to velodyne transform
        if self.T_base_velo is None:
            self.T_base_velo = np.eye(4)
            
        T_velo_base = np.linalg.inv(self.T_base_velo)

        # 2. If first frame, initialize global pose
        if self.prev_pcd is None:
            self.prev_pcd = pcd_down
            self.T_world_base = np.eye(4)
            
            # Reset relative IMU integration
            self.imu_rel_t = np.zeros(3)
            self.imu_rel_v = np.zeros(3)
            self.imu_rel_q = np.array([0.0, 0.0, 0.0, 1.0])
            
            return self.get_odom_msg(header)

        # 3. Create initial guess from IMU prediction
        T_base_rel = tf.transformations.quaternion_matrix(self.imu_rel_q)
        T_base_rel[:3, 3] = self.imu_rel_t
        
        # Convert base_link relative transform to LiDAR (velodyne) frame
        T_velo_rel = T_velo_base @ T_base_rel @ self.T_base_velo
        
        # 4. Perform Open3D ICP Scan Matching
        init_guess = T_velo_rel
        
        try:
            reg = o3d.pipelines.registration.registration_icp(
                pcd_down, self.prev_pcd, self.max_distance, init_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
            )
            
            # Use refined transform if fitness is good, else fall back to IMU guess
            if reg.fitness > 0.4:
                T_refined_velo = reg.transformation
            else:
                T_refined_velo = T_velo_rel
                rospy.logwarn_throttle(5, f"ICP fitness too low ({reg.fitness:.2f}), using IMU prediction.")
        except Exception as e:
            rospy.logerr_throttle(5, f"ICP matching failed: {e}. Falling back to IMU.")
            T_refined_velo = T_velo_rel

        # 5. Update global pose
        T_world_velo_old = self.T_world_base @ self.T_base_velo
        T_world_velo_new = T_world_velo_old @ T_refined_velo
        
        # Convert back to base_link global pose
        self.T_world_base = T_world_velo_new @ T_velo_base
        
        # 6. Save current scan for next iteration
        self.prev_pcd = pcd_down
        
        # Reset relative IMU integration for the next interval
        self.imu_rel_t = np.zeros(3)
        self.imu_rel_v = np.zeros(3)
        self.imu_rel_q = np.array([0.0, 0.0, 0.0, 1.0])

        return self.get_odom_msg(header)

    def get_odom_msg(self, header):
        odom = Odometry()
        odom.header.stamp = header.stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        
        pos = self.T_world_base[:3, 3]
        q = tf.transformations.quaternion_from_matrix(self.T_world_base)
        
        odom.pose.pose.position = Point(pos[0], pos[1], pos[2])
        odom.pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
        
        return odom
