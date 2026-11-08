#!/usr/bin/env python3
"""
arena_builder.py — Procedural Continuous Pac-Man Arena Builder for MuJoCo.

Generates continuous 2D procedural mazes using pacmanbot.world.World and compiles
them directly into native MuJoCo MjModel instances via MjSpec:
  • Continuous capsule walls with authentic collision thickness
  • White regular pellets (10 pts) and golden emissive power pellets (50 pts)
  • SvanM2 quadruped robots attached via MjSpec.attach with distinct ghost prefixes
  • Pac-Man vehicle agent
"""

import sys
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import mujoco

# Ensure pacmanbot is in path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
if str(_PACMANBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PACMANBOT_DIR))

from world import World, Capsule, CurvedWall
from curriculum import STAGES

SVANM2_XML_PATH = _REPO_ROOT / "xterra_mjlab" / "assets" / "svanm2" / "xml" / "svanm2_mjlab.xml"

# Authentic Pac-Man Ghost body colors (RGB alpha)
GHOST_COLORS = [
    (0.96, 0.12, 0.12, 1.0),  # Ghost 0: Red (Blinky)
    (1.00, 0.52, 0.75, 1.0),  # Ghost 1: Pink (Pinky)
    (0.00, 0.90, 0.98, 1.0),  # Ghost 2: Cyan (Inky)
    (1.00, 0.58, 0.10, 1.0),  # Ghost 3: Orange (Clyde)
    (0.68, 0.22, 0.98, 1.0),  # Ghost 4: Purple (Sue)
    (0.18, 0.95, 0.25, 1.0),  # Ghost 5: Green (Funky)
    (1.00, 0.84, 0.05, 1.0),  # Ghost 6: Gold (Spunky)
]


def liang_barsky_clip(x1: float, y1: float, x2: float, y2: float,
                      xmin: float, xmax: float, ymin: float, ymax: float):
    """Clips a 2D line segment (x1, y1) -> (x2, y2) to the bounding box [xmin, xmax] x [ymin, ymax]."""
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    if u1 > u2:
        return None
    cx1 = x1 + u1 * dx
    cy1 = y1 + u1 * dy
    cx2 = x1 + u2 * dx
    cy2 = y1 + u2 * dy
    if (cx2 - cx1)**2 + (cy2 - cy1)**2 < 0.04:
        return None
    return cx1, cy1, cx2, cy2


class ContinuousArena:
    """Represents a compiled procedural MuJoCo arena with game entities."""

    def __init__(
        self,
        stage: int = 1,
        scale: float = 1.5,
        seed: int = None,
        wall_height: float = 0.75,
        world: Any = None,
        grid: Any = None,
        player_start_grid: Tuple[float, float] = None,
        ghost_starts_grid: List[Tuple[float, float]] = None,
    ):
        self.stage_idx = stage
        self.scale = scale
        self.wall_height = wall_height

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Retrieve curriculum stage configuration (0-indexed internally)
        stage_cfg_idx = max(0, min(len(STAGES) - 1, stage - 1))
        stage_cfg = STAGES[stage_cfg_idx]
        self.raw_w = stage_cfg.cols
        self.raw_h = stage_cfg.rows
        self.n_ghosts = stage_cfg.n_ghosts
        self.n_power = stage_cfg.n_power
        self.pac_speed = stage_cfg.pac_speed

        self.width_m = self.raw_w * self.scale
        self.height_m = self.raw_h * self.scale
        self.half_w = self.width_m / 2.0
        self.half_h = self.height_m / 2.0

        # Origin offset to center arena at (0, 0) in world coordinates
        self.x_offset = self.raw_w / 2.0
        self.y_offset = self.raw_h / 2.0

        if world is not None and grid is not None and player_start_grid is not None and ghost_starts_grid is not None:
            self.world = world
            self.grid = grid
            self.player_start_grid = player_start_grid
            self.ghost_starts_grid = ghost_starts_grid
        else:
            # Generate procedural continuous maze using pacmanbot exactly matching pacman.py
            from pacman import generate_map
            self.grid, self.player_start_grid, self.world = generate_map(
                world_height=float(self.raw_h),
                world_width=float(self.raw_w),
                n_power=self.n_power,
                random_spawn=False,
                obs_resolution=1.0,
            )

            # Compute ghost starting locations matching pacman.py:new_game()
            open_cells_arr = None
            if hasattr(self.world, 'prm_nodes') and self.world.prm_nodes:
                open_cells_arr = np.array(self.world.prm_nodes)  # shape (N, 2): (y, x)
            if open_cells_arr is None or len(open_cells_arr) < self.n_ghosts:
                safe = getattr(self.world, 'safe_area', [])
                if safe:
                    sampled = random.sample(safe, min(len(safe), max(self.n_ghosts * 3, 20)))
                    open_cells_arr = np.array([(pt[1], pt[0]) for pt in sampled])
                else:
                    open_cells_arr = np.array([[float(self.player_start_grid[0]), float(self.player_start_grid[1])]])

            pac_yx = np.array([float(self.player_start_grid[0]), float(self.player_start_grid[1])])
            d_pac = np.sum(np.square(open_cells_arr - pac_yx), axis=1)
            avail = np.ones(len(open_cells_arr), dtype=bool)
            first_idx = int(np.argmax(d_pac))
            self.ghost_starts_grid = [tuple(open_cells_arr[first_idx])]
            avail[first_idx] = False
            min_dist_to_ghosts = np.full(len(open_cells_arr), np.inf)
            for _ in range(self.n_ghosts - 1):
                last_placed = np.array(self.ghost_starts_grid[-1])
                dist_to_last = np.sum(np.square(open_cells_arr - last_placed), axis=1)
                min_dist_to_ghosts = np.minimum(min_dist_to_ghosts, dist_to_last)
                scores = np.minimum(d_pac, min_dist_to_ghosts)
                scores[~avail] = -1
                best_idx = int(np.argmax(scores))
                self.ghost_starts_grid.append(tuple(open_cells_arr[best_idx]))
                avail[best_idx] = False

        # Convert grid coordinates (row, col) = (y, x) to MuJoCo continuous metric coordinates
        self.pacman_start = self.to_world(self.player_start_grid[1], self.player_start_grid[0])
        self.ghost_starts = [self.to_world(gs[1], gs[0]) for gs in self.ghost_starts_grid]

        # Collect pellet coordinates in metric space and retain grid references
        self.pellets: Dict[Any, Tuple[float, float]] = {}
        self.pellet_grid_pts: Dict[Any, Tuple[float, float]] = {}
        for idx, (px, py) in enumerate(self.world.pellets):
            wx, wy = self.to_world(px, py)
            self.pellets[idx] = (wx, wy)
            self.pellet_grid_pts[idx] = (px, py)

        self.power_pellets: Dict[Any, Tuple[float, float]] = {}
        self.power_grid_pts: Dict[Any, Tuple[float, float]] = {}
        for idx, (ppx, ppy) in enumerate(self.world.power_pellets):
            wx, wy = self.to_world(ppx, ppy)
            self.power_pellets[idx] = (wx, wy)
            self.power_grid_pts[idx] = (ppx, ppy)

        # Build and compile MuJoCo MjSpec
        self.model, self.data = self._build_mujoco_model()

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pacmanbot continuous coordinates to metric MuJoCo world space."""
        x = (float(px) - self.x_offset) * self.scale
        y = (float(py) - self.y_offset) * self.scale
        return (float(x), float(y))

    def to_grid(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert metric MuJoCo world space coordinates back to pacmanbot continuous coordinates."""
        px = (float(wx) / self.scale) + self.x_offset
        py = (float(wy) / self.scale) + self.y_offset
        return (float(px), float(py))

    def _build_mujoco_model(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Constructs and compiles the composite MuJoCo model."""
        hw, hh = self.half_w, self.half_h
        hz = self.wall_height / 2.0
        r_wall = 0.5 * self.scale

        xml_chunks = [
            f"<mujoco model='pacman_continuous_stage_{self.stage_idx}'>",
            "  <compiler angle='radian' autolimits='true'/>",
            "  <option timestep='0.002' gravity='0 0 -9.81' cone='elliptic' impratio='100'/>",
            "  <visual>",
            "    <headlight ambient='0.42 0.42 0.48' diffuse='0.60 0.60 0.65' specular='0.2 0.2 0.2'/>",
            "    <rgba haze='0.05 0.05 0.08 1.0'/>",
            "    <global offwidth='1920' offheight='1080'/>",
            "  </visual>",
            "  <asset>",
            "    <material name='wall_bnd' rgba='0.05 0.55 1.0 1.0' emission='0.50' specular='0.5'/>",
            "    <material name='wall_inner' rgba='0.62 0.18 0.95 1.0' emission='0.38' specular='0.4'/>",
            "    <material name='floor_mat' rgba='0.07 0.08 0.12 1.0' specular='0.25' reflectance='0.05'/>",
            "    <material name='pellet_mat' rgba='1.0 0.98 0.92 1.0' emission='0.80' specular='0.3'/>",
            "    <material name='power_mat' rgba='1.0 0.85 0.05 1.0' emission='1.0' specular='0.6'/>",
            "    <material name='defused_mat' rgba='0.35 0.85 1.0 0.9' emission='0.45' specular='0.5'/>",
            "    <material name='pacman_chassis_mat' rgba='0.98 0.78 0.02 1.0' emission='0.40' specular='0.8'/>",
            "    <material name='pacman_dark_mat' rgba='0.12 0.12 0.14 1.0' specular='0.4'/>",
            "    <material name='pacman_sensor_mat' rgba='0.02 0.02 0.04 1.0' specular='0.9'/>",
            "    <material name='pacman_lens_mat' rgba='0.0 0.85 1.0 1.0' emission='0.85'/>",
            "    <material name='pacman_headlight_mat' rgba='1.0 0.98 0.85 1.0' emission='0.95'/>",
            "  </asset>",
            "  <worldbody>",
            f"    <geom name='floor' type='plane' size='{hw + 6.0} {hh + 6.0} 0.1' material='floor_mat'/>",
            f"    <light pos='{hw*0.5:.2f} {hh*0.5:.2f} 18' dir='0 0 -1' diffuse='0.65 0.65 0.75' specular='0.15 0.15 0.15'/>",
            f"    <light pos='{-hw*0.5:.2f} {hh*0.5:.2f} 18' dir='0 0 -1' diffuse='0.65 0.65 0.75' specular='0.15 0.15 0.15'/>",
            f"    <light pos='{hw*0.5:.2f} {-hh*0.5:.2f} 18' dir='0 0 -1' diffuse='0.65 0.65 0.75' specular='0.15 0.15 0.15'/>",
            f"    <light pos='{-hw*0.5:.2f} {-hh*0.5:.2f} 18' dir='0 0 -1' diffuse='0.65 0.65 0.75' specular='0.15 0.15 0.15'/>",
            f"    <light pos='0 0 22' dir='0 0 -1' diffuse='0.85 0.85 0.95' specular='0.25 0.25 0.25'/>",
        ]

        # Add arena perimeter boundary walls (Electric Blue Neon)
        xml_chunks.append(f"    <geom name='bnd_south' type='capsule' fromto='{-hw} {-hh} {hz} {hw} {-hh} {hz}' size='{r_wall}' material='wall_bnd'/>")
        xml_chunks.append(f"    <geom name='bnd_north' type='capsule' fromto='{-hw} {hh} {hz} {hw} {hh} {hz}' size='{r_wall}' material='wall_bnd'/>")
        xml_chunks.append(f"    <geom name='bnd_west' type='capsule' fromto='{-hw} {-hh} {hz} {-hw} {hh} {hz}' size='{r_wall}' material='wall_bnd'/>")
        xml_chunks.append(f"    <geom name='bnd_east' type='capsule' fromto='{hw} {-hh} {hz} {hw} {hh} {hz}' size='{r_wall}' material='wall_bnd'/>")

        # Extract internal obstacles and clip strictly to arena bounds (Cyberpunk Magenta/Purple)
        internal_caps = []
        for idx, obs in enumerate(self.world.obstacles):
            sub_caps = [obs] if isinstance(obs, Capsule) else obs.capsules
            if idx >= 4:
                internal_caps.extend(sub_caps)

        xmin, xmax = 0.5, float(self.raw_w) - 0.5
        ymin, ymax = 0.5, float(self.raw_h) - 0.5

        capsule_count = 0
        for c in internal_caps:
            res = liang_barsky_clip(c.x1, c.y1, c.x2, c.y2, xmin, xmax, ymin, ymax)
            if res is not None:
                wx1, wy1 = self.to_world(res[0], res[1])
                wx2, wy2 = self.to_world(res[2], res[3])
                r_c = c.r * self.scale
                if math.hypot(wx2 - wx1, wy2 - wy1) > 0.05:
                    xml_chunks.append(
                        f"    <geom name='wall_{capsule_count}' type='capsule' "
                        f"fromto='{wx1:.4f} {wy1:.4f} {hz:.4f} {wx2:.4f} {wy2:.4f} {hz:.4f}' "
                        f"size='{r_c:.4f}' material='wall_inner'/>"
                    )
                    capsule_count += 1

        # Add regular pellets (white glowing pearls, contype=0)
        pellet_z = 0.35
        for idx, (px, py) in self.pellets.items():
            xml_chunks.append(
                f"    <geom name='pellet_{idx}' type='sphere' pos='{px:.4f} {py:.4f} {pellet_z}' "
                f"size='0.09' material='pellet_mat' contype='0' conaffinity='0'/>"
            )

        # Add power pellets (radiant golden orbs, contype=0)
        power_z = 0.40
        for idx, (ppx, ppy) in self.power_pellets.items():
            xml_chunks.append(
                f"    <geom name='power_{idx}' type='sphere' pos='{ppx:.4f} {ppy:.4f} {power_z}' "
                f"size='0.20' material='power_mat' contype='0' conaffinity='0'/>"
            )

        # Add Pac-Man Agent (Autonomous Rover Car with 4 Mecanum Wheels)
        pac_x, pac_y = self.pacman_start
        xml_chunks.extend([
            f"    <body name='pacman' pos='{pac_x:.4f} {pac_y:.4f} 0.18'>",
            "      <freejoint name='pacman_root'/>",
            "      <!-- Main car chassis body in radiant bumblebee gold / electric cyan -->",
            "      <geom name='pacman_chassis' type='box' pos='0 0 0.02' size='0.22 0.14 0.045' material='pacman_chassis_mat' contype='0' conaffinity='0'/>",
            "      <!-- Aerodynamic cockpit dome / windshield canopy -->",
            "      <geom name='pacman_dome' type='box' pos='-0.02 0 0.075' size='0.12 0.085 0.035' material='pacman_sensor_mat' contype='0' conaffinity='0'/>",
            "      <!-- Front hood and aerodynamic front/rear bumpers -->",
            "      <geom name='pacman_hood' type='box' pos='0.15 0 0.005' size='0.06 0.11 0.025' material='pacman_chassis_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_bumper_f' type='box' pos='0.215 0 -0.015' size='0.018 0.145 0.025' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_bumper_r' type='box' pos='-0.215 0 -0.015' size='0.018 0.145 0.025' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <!-- Top-mounted 3D LiDAR puck with glowing optical window -->",
            "      <geom name='pacman_lidar_base' type='cylinder' pos='-0.02 0 0.125' size='0.045 0.018' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_lidar_lens' type='cylinder' pos='-0.02 0 0.136' size='0.048 0.007' material='pacman_lens_mat' contype='0' conaffinity='0'/>",
            "      <!-- Front twin LED headlights -->",
            "      <geom name='pacman_headlight_l' type='sphere' pos='0.22 0.09 0.00' size='0.022' material='pacman_headlight_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_headlight_r' type='sphere' pos='0.22 -0.09 0.00' size='0.022' material='pacman_headlight_mat' contype='0' conaffinity='0'/>",
            "      <!-- Rear twin LED taillights -->",
            "      <geom name='pacman_taillight_l' type='sphere' pos='-0.22 0.09 0.00' size='0.018' material='wall_inner' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_taillight_r' type='sphere' pos='-0.22 -0.09 0.00' size='0.018' material='wall_inner' contype='0' conaffinity='0'/>",
            "      <!-- Four Mecanum Wheels at chassis corners with metallic wheel hubs -->",
            "      <geom name='pacman_wheel_fl' type='cylinder' pos='0.13 0.17 -0.035' quat='0.7071 0.7071 0 0' size='0.065 0.024' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_wheel_fr' type='cylinder' pos='0.13 -0.17 -0.035' quat='0.7071 0.7071 0 0' size='0.065 0.024' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_wheel_rl' type='cylinder' pos='-0.13 0.17 -0.035' quat='0.7071 0.7071 0 0' size='0.065 0.024' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_wheel_rr' type='cylinder' pos='-0.13 -0.17 -0.035' quat='0.7071 0.7071 0 0' size='0.065 0.024' material='pacman_dark_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_hub_l' type='cylinder' pos='0.13 0.186 -0.035' quat='0.7071 0.7071 0 0' size='0.032 0.010' material='pacman_chassis_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_hub_r' type='cylinder' pos='0.13 -0.186 -0.035' quat='0.7071 0.7071 0 0' size='0.032 0.010' material='pacman_chassis_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_hub_rl' type='cylinder' pos='-0.13 0.186 -0.035' quat='0.7071 0.7071 0 0' size='0.032 0.010' material='pacman_chassis_mat' contype='0' conaffinity='0'/>",
            "      <geom name='pacman_hub_rr' type='cylinder' pos='-0.13 -0.186 -0.035' quat='0.7071 0.7071 0 0' size='0.032 0.010' material='pacman_chassis_mat' contype='0' conaffinity='0'/>",
            "    </body>",
            "  </worldbody>",
            "</mujoco>",
        ])

        arena_xml = "\n".join(xml_chunks)
        spec = mujoco.MjSpec.from_string(arena_xml)

        # Attach SvanM2 quadruped robots with authentic ghost body colors (no hovering LED)
        assert SVANM2_XML_PATH.exists(), f"SvanM2 XML missing at {SVANM2_XML_PATH}"

        for gid in range(self.n_ghosts):
            gx, gy = self.ghost_starts[gid]
            f = spec.worldbody.add_frame()
            f.pos = [gx, gy, 0.34]

            robot_spec = mujoco.MjSpec.from_file(str(SVANM2_XML_PATH))

            # Apply ghost signature color directly to SvanM2 quadruped torso body parts
            col = GHOST_COLORS[gid % len(GHOST_COLORS)]
            t_idx = 0
            for g in robot_spec.geoms:
                if g.meshname == "torso":
                    g.rgba = [col[0], col[1], col[2], 1.0]
                    g.name = f"torso_{t_idx}"
                    t_idx += 1
                elif "hip" in (g.meshname or ""):
                    g.rgba = [0.22, 0.22, 0.25, 1.0]

            prefix = f"ghost_{gid}/"
            spec.attach(robot_spec, prefix=prefix, frame=f)

        model = spec.compile()
        data = mujoco.MjData(model)

        return model, data
