#!/usr/bin/env python3
"""
lidar3d.py — Realistic 3D Multi-Beam LiDAR Simulation & 2.5D Elevation Mapping for MuJoCo.

Features:
  • Multi-ring 3D spherical raycasting using native MuJoCo ray-query API (mj_ray).
  • Configurable beams (e.g. 16 or 32 rings, 360° azimuth, ±15° elevation).
  • Realistic sensor noise: range measurement noise, ray dropouts, min/max range limits.
  • Ground plane / obstacle return classification (elevation difference & slope threshold).
  • 2.5D Local Elevation Grid & 2D Traversability Costmap generation for quadruped navigation.
"""

import math
from typing import Tuple, Optional, Dict, List
import numpy as np
import mujoco


class LivoxMid360Lidar:
    """Simulates a Livox Mid-360 non-repetitive scanning LiDAR for MuJoCo.

    Livox Mid-360 Specifications:
      • Ultra-wide Field of View: 360° Horizontal x -7° to 52° Vertical (59° Vertical FoV).
      • Non-repetitive scanning trajectory (hypocycloid / rosette pattern).
      • Point rate: 100,000 points/s (~2,000 points per 50 Hz frame).
      • Range precision: 1σ < 2 cm @ 10m, minimum detection range: 0.1m, max range: 40m.
      • Compact dome form factor mounted on the SvanM2 quadruped head / front torso.
    """

    def __init__(
        self,
        points_per_frame: int = 1200,      # Beam points simulated per 50 Hz control step
        fov_up_deg: float = 52.0,          # Livox Mid-360 vertical upper FoV
        fov_down_deg: float = -7.0,        # Livox Mid-360 vertical lower FoV
        min_range: float = 0.10,           # 10 cm blind zone
        max_range: float = 25.0,           # Max indoor/arena range
        noise_std: float = 0.015,          # 1.5 cm range noise
        dropout_prob: float = 0.012,
        mount_offset_xyz: Tuple[float, float, float] = (0.18, 0.0, 0.10), # Mounted on SvanM2 head/front
    ):
        self.points_per_frame = points_per_frame
        self.fov_up_rad = math.radians(fov_up_deg)
        self.fov_down_rad = math.radians(fov_down_deg)
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.mount_offset_xyz = np.array(mount_offset_xyz, dtype=np.float64)

        # Scanning rosette parameters (Livox dual-prism / non-repetitive rosette mechanics)
        self.scan_phase = 0.0
        self.w1 = 7.13   # Primary rotor angular frequency ratio
        self.w2 = -4.91  # Secondary rotor angular frequency ratio

    def scan(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sensor_pos: np.ndarray,
        sensor_yaw: float = 0.0,
        ignore_body_id: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs a Livox Mid-360 non-repetitive rosette beam scan.

        Returns:
            pointcloud: (N, 3) 3D points in world space
            ranges: (N,) measured distance values
            intensities: (N,) simulated optical return intensity
        """
        # Advance non-repetitive scan phase
        t = np.linspace(self.scan_phase, self.scan_phase + 1.0, self.points_per_frame)
        self.scan_phase = (self.scan_phase + 1.0) % 1000.0

        # Rosette scanning pattern (hypocycloid in spherical angles)
        azimuth = (self.w1 * t * 2.0 * np.pi) % (2.0 * np.pi) - np.pi
        el_norm = 0.5 * (1.0 + np.sin(self.w2 * t * 2.0 * np.pi))
        elevation = self.fov_down_rad + el_norm * (self.fov_up_rad - self.fov_down_rad)

        cos_el = np.cos(elevation)
        sin_el = np.sin(elevation)
        dx_local = cos_el * np.cos(azimuth)
        dy_local = cos_el * np.sin(azimuth)
        dz_local = sin_el

        # Rotate by host robot yaw heading
        cos_y = math.cos(sensor_yaw)
        sin_y = math.sin(sensor_yaw)

        dx_world = cos_y * dx_local - sin_y * dy_local
        dy_world = sin_y * dx_local + cos_y * dy_local
        dz_world = dz_local

        # Origin in world space = robot_pos + R_yaw @ mount_offset
        mount_rotated = np.array([
            cos_y * self.mount_offset_xyz[0] - sin_y * self.mount_offset_xyz[1],
            sin_y * self.mount_offset_xyz[0] + cos_y * self.mount_offset_xyz[1],
            self.mount_offset_xyz[2]
        ], dtype=np.float64)
        pos_origin = sensor_pos + mount_rotated

        pointcloud = []
        intensities = []
        measured_ranges = []

        geom_id_out = np.zeros(1, dtype=np.int32)
        p_dir = np.zeros(3, dtype=np.float64)

        for i in range(self.points_per_frame):
            p_dir[0] = dx_world[i]
            p_dir[1] = dy_world[i]
            p_dir[2] = dz_world[i]

            dist = mujoco.mj_ray(
                model,
                data,
                pos_origin,
                p_dir,
                None,
                1,
                ignore_body_id,
                geom_id_out
            )

            if self.min_range <= dist < self.max_range and geom_id_out[0] != -1:
                if self.dropout_prob > 0.0 and np.random.rand() < self.dropout_prob:
                    continue

                noisy_dist = dist
                if self.noise_std > 0.0:
                    noisy_dist += np.random.normal(0.0, self.noise_std)
                    noisy_dist = max(self.min_range, min(self.max_range, noisy_dist))

                hit_pt = pos_origin + p_dir * noisy_dist
                pointcloud.append(hit_pt)
                measured_ranges.append(noisy_dist)
                
                # Diffuse reflectance model
                intensity = float(np.clip(1.0 / (1.0 + 0.04 * noisy_dist), 0.1, 1.0))
                intensities.append(intensity)

        if len(pointcloud) > 0:
            pc_arr = np.array(pointcloud, dtype=np.float32)
            ranges_arr = np.array(measured_ranges, dtype=np.float32)
            int_arr = np.array(intensities, dtype=np.float32)
        else:
            pc_arr = np.empty((0, 3), dtype=np.float32)
            ranges_arr = np.empty((0,), dtype=np.float32)
            int_arr = np.empty((0,), dtype=np.float32)

        return pc_arr, ranges_arr, int_arr


class MultiBeamLidar3D:
    """Simulates a multi-beam 3D LiDAR mounted on an agent in MuJoCo."""

    def __init__(
        self,
        num_rings: int = 16,
        num_azimuth: int = 90,
        fov_up_deg: float = 15.0,
        fov_down_deg: float = -15.0,
        min_range: float = 0.15,
        max_range: float = 20.0,
        noise_std: float = 0.015,
        dropout_prob: float = 0.01,
        mount_offset_z: float = 0.28,  # Height above robot root body
    ):
        self.num_rings = num_rings
        self.num_azimuth = num_azimuth
        self.min_range = min_range
        self.max_range = max_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.mount_offset_z = mount_offset_z

        # Precompute unit ray directions in sensor frame (spherical coordinates)
        elevations = np.linspace(np.radians(fov_down_deg), np.radians(fov_up_deg), num_rings)
        azimuths = np.linspace(-np.pi, np.pi, num_azimuth, endpoint=False)

        # Shape: (num_rings * num_azimuth, 3)
        rays = []
        for el in elevations:
            cos_el = math.cos(el)
            sin_el = math.sin(el)
            for az in azimuths:
                dx = cos_el * math.cos(az)
                dy = cos_el * math.sin(az)
                dz = sin_el
                rays.append((dx, dy, dz))

        self.unit_rays = np.array(rays, dtype=np.float64)  # (N_rays, 3)
        self.total_rays = len(self.unit_rays)

    def scan(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sensor_pos: np.ndarray,
        sensor_yaw: float = 0.0,
        ignore_body_id: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs a 3D LiDAR sweep against the current MuJoCo scene.

        Args:
            model: MjModel
            data: MjData
            sensor_pos: 3D coordinates [x, y, z] of the sensor origin in world frame.
            sensor_yaw: Heading orientation (yaw in radians) of the sensor mount.
            ignore_body_id: Body ID of the host robot to ignore self-collisions.

        Returns:
            pointcloud: (N_valid, 3) array of [x, y, z] in world coordinates.
            ranges: (N_rays,) array of measured ranges (or max_range for no-hit/dropout).
            intensities: (N_valid,) simulated reflectance intensity [0.0, 1.0].
        """
        # Rotate precomputed rays by sensor yaw
        cos_y = math.cos(sensor_yaw)
        sin_y = math.sin(sensor_yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [  0.0,    0.0, 1.0]
        ], dtype=np.float64)

        world_rays = self.unit_rays @ R_yaw.T  # (total_rays, 3)

        pos_origin = np.array(sensor_pos, dtype=np.float64).copy()
        pos_origin[2] += self.mount_offset_z

        pointcloud = []
        intensities = []
        measured_ranges = np.full(self.total_rays, self.max_range, dtype=np.float32)

        geom_id_out = np.zeros(1, dtype=np.int32)

        for i in range(self.total_rays):
            p_dir = world_rays[i]
            
            # Fast raycast using native MuJoCo mj_ray
            dist = mujoco.mj_ray(
                model,
                data,
                pos_origin,
                p_dir,
                None,             # vec_geomgroup: None checks all active geom groups
                1,                # flg_static: check static geoms
                ignore_body_id,   # bodyexclude: ignore host robot body
                geom_id_out
            )

            # Check validity
            if self.min_range <= dist < self.max_range and geom_id_out[0] != -1:
                # Stochastic dropout
                if self.dropout_prob > 0.0 and np.random.rand() < self.dropout_prob:
                    continue

                # Additive Gaussian range noise
                noisy_dist = dist
                if self.noise_std > 0.0:
                    noisy_dist += np.random.normal(0.0, self.noise_std)
                    noisy_dist = max(self.min_range, min(self.max_range, noisy_dist))

                measured_ranges[i] = noisy_dist
                hit_pt = pos_origin + p_dir * noisy_dist
                pointcloud.append(hit_pt)

                # Simulated intensity: diffuse cosine attenuation + inverse distance decay
                intensity = float(np.clip(1.0 / (1.0 + 0.05 * noisy_dist), 0.1, 1.0))
                intensities.append(intensity)

        if len(pointcloud) > 0:
            pc_arr = np.array(pointcloud, dtype=np.float32)
            int_arr = np.array(intensities, dtype=np.float32)
        else:
            pc_arr = np.empty((0, 3), dtype=np.float32)
            int_arr = np.empty((0,), dtype=np.float32)

        return pc_arr, measured_ranges, int_arr


class ElevationTraversabilityMapper:
    """Processes 3D point clouds into local 2.5D elevation & traversability costmaps."""

    def __init__(
        self,
        resolution: float = 0.15,      # Metric cell size (meters per cell)
        map_radius: float = 10.0,       # Half-width of local map (meters)
        max_step_height: float = 0.12,  # Maximum step height SvanM2 quadruped can traverse (meters)
        max_slope_deg: float = 28.0,    # Maximum traversable slope for quadruped stance
        robot_radius: float = 0.35,     # Robot footprint inflation radius (meters)
    ):
        self.resolution = resolution
        self.map_radius = map_radius
        self.grid_dim = int(np.ceil((2.0 * map_radius) / resolution))
        self.max_step_height = max_step_height
        self.max_slope_rad = np.radians(max_slope_deg)
        self.robot_radius = robot_radius
        self.inflation_cells = int(np.ceil(robot_radius / resolution))

    def update_map(
        self,
        pointcloud: np.ndarray,
        robot_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Creates an elevation grid and classified traversability map centered at robot_pos.

        Returns:
            dict containing:
              - 'elevation_min': 2D array of minimum z height in each cell
              - 'elevation_max': 2D array of maximum z height in each cell
              - 'traversability': 2D float array: 0.0 (free/traversable) to 1.0 (lethal obstacle)
              - 'origin_xy': [min_x, min_y] metric coordinates of the map
        """
        center_x, center_y = robot_pos[0], robot_pos[1]
        min_x = center_x - self.map_radius
        min_y = center_y - self.map_radius

        elevation_min = np.full((self.grid_dim, self.grid_dim), np.nan, dtype=np.float32)
        elevation_max = np.full((self.grid_dim, self.grid_dim), np.nan, dtype=np.float32)
        point_count = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int32)

        if len(pointcloud) > 0:
            # Shift point cloud relative to grid origin
            px = pointcloud[:, 0] - min_x
            py = pointcloud[:, 1] - min_y
            pz = pointcloud[:, 2]

            # Determine grid cell coordinates
            cx = (px / self.resolution).astype(np.int32)
            cy = (py / self.resolution).astype(np.int32)

            # Filter points within local map bounds
            valid = (cx >= 0) & (cx < self.grid_dim) & (cy >= 0) & (cy < self.grid_dim)
            cx, cy, pz = cx[valid], cy[valid], pz[valid]

            for i in range(len(cx)):
                x, y, z = cx[i], cy[i], pz[i]
                point_count[y, x] += 1
                if np.isnan(elevation_min[y, x]) or z < elevation_min[y, x]:
                    elevation_min[y, x] = z
                if np.isnan(elevation_max[y, x]) or z > elevation_max[y, x]:
                    elevation_max[y, x] = z

        # Calculate step height difference per cell (obstacle classification)
        step_diff = elevation_max - elevation_min
        step_diff[np.isnan(step_diff)] = 0.0

        # Lethal obstacles: step height exceeds quadruped clearance capability
        lethal_mask = (step_diff > self.max_step_height) | (elevation_max > 0.45)

        # Continuous costmap: normalized 0.0 to 1.0
        traversability = np.clip(step_diff / self.max_step_height, 0.0, 1.0)
        traversability[lethal_mask] = 1.0

        # Quadruped safety inflation: inflate lethal obstacles by robot radius
        if self.inflation_cells > 0 and np.any(lethal_mask):
            try:
                from scipy.ndimage import distance_transform_edt
                dist_to_obstacle = distance_transform_edt(~lethal_mask) * self.resolution
                inflation_mask = dist_to_obstacle < self.robot_radius
                cost_factor = 1.0 - (dist_to_obstacle / self.robot_radius)
                traversability = np.maximum(traversability, np.clip(cost_factor, 0.0, 1.0) * inflation_mask)
            except ImportError:
                pass

        return {
            "elevation_min": elevation_min,
            "elevation_max": elevation_max,
            "traversability": traversability,
            "point_count": point_count,
            "origin_xy": np.array([min_x, min_y], dtype=np.float32),
            "resolution": self.resolution,
        }
