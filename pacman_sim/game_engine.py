#!/usr/bin/env python3
"""
game_engine.py — Continuous Simulation Game Engine & Referee for MuJoCo.

Coordinates the continuous simulation loop:
  • Pac-Man autonomous navigation (Player agent from pacmanbot)
  • Pellet collection & power pellet consumption with 3D geom updates
  • Power pellet defusal (CONVERT rule) by ghost swarm
  • Contact referee: 0.55m pellet pickup, 0.50m agent collision
  • Strict fog-of-war: ghost power awareness ONLY updated through direct line-of-sight
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import mujoco
from collections import deque
import heapq
import pathfinder
from pacman import (
    Player,
    _NUM_STEERING_RAYS,
    _STEERING_RAY_VX,
    _STEERING_RAY_VY,
    _STEERING_DX,
    _STEERING_DY,
    _STEERING_N_STEPS,
    _STEERING_CHECK_DIST,
)
from arena_builder import ContinuousArena, GHOST_COLORS
from svanm2_runner import SvanM2Policy, SvanM2Runner
from hive_coordinator import HiveCoordinator


class ContinuousPlayer(Player):
    """Adapts pacmanbot.pacman.Player for 50 Hz continuous physical simulation."""

    def __init__(
        self,
        grid: np.ndarray,
        pos: Tuple[float, float],
        world,
        scale: float = 1.5,
        target_speed_metric: float = 1.35,
    ):
        super().__init__(grid=grid, pos=pos, world=world)
        self.scale = scale
        # Physical collision radius in grid units:
        # Pac-Man chassis is 0.22m cylinder + 0.015m bumper = 0.235m.
        # At scale 1.5 m/unit, 0.235m / 1.5 = 0.157 grid units.
        # Setting radius = 0.16 prevents doorway locking while matching physical geometry.
        self.radius = 0.16
        self.base_speed_metric = target_speed_metric
        self.max_speed = 1.0  # Keep 1.0 for normalized steering & full 1.8 unit lookahead
        self.replan_age = 25   # 0.50s at 50 Hz (balanced responsiveness without route thrashing)
        self.emergency_dist = 2.5
        self.pos_check_timer = 0
        self.pos_history = deque(maxlen=6)
        self.steer_vx = 0.0
        self.steer_vy = 0.0
        self.path_this_frame: List[Tuple[float, float]] = []
        self.manual_mode = False
        self.manual_vx = 0.0
        self.manual_vy = 0.0

    def step_continuous(
        self,
        dt: float,
        ghosts: dict,
        powered: bool,
    ) -> Tuple[float, float, float, float]:
        """
        Runs PRM pathfinding and 16-ray context steering, advancing Pac-Man continuously.
        Returns: (pac_wx, pac_wy, pac_vx_world, pac_vy_world)
        """
        self.frame_counter += 1
        self._route_age += 1
        self.powered = powered

        # 0. Manual keyboard driving mode (WASD / Arrow Keys)
        if self.manual_mode:
            unit_vx = float(self.manual_vx)
            unit_vy = float(self.manual_vy)
            mag = math.hypot(unit_vx, unit_vy)
            if mag > 1.0:
                unit_vx /= mag
                unit_vy /= mag

            spd_mult = 1.15 if self.powered else 1.0
            cur_speed_metric = self.base_speed_metric * spd_mult
            v_grid_s = cur_speed_metric / self.scale
            prev_wx = float((self.x - self.world.width / 2.0) * self.scale)
            prev_wy = float((self.y - self.world.height / 2.0) * self.scale)

            step_dist = v_grid_s * dt
            if mag > 0.05:
                self.x += unit_vx * step_dist
                self.y += unit_vy * step_dist
                self.x, self.y = self.world.resolve_collision(self.x, self.y, self.radius, max_iters=3)

            curr_wx = float((self.x - self.world.width / 2.0) * self.scale)
            curr_wy = float((self.y - self.world.height / 2.0) * self.scale)
            self.path_this_frame = [(prev_wx, prev_wy), (curr_wx, curr_wy)]
            eff_vx_world = (unit_vx * cur_speed_metric) if mag > 0.05 else 0.0
            eff_vy_world = (unit_vy * cur_speed_metric) if mag > 0.05 else 0.0
            self.vx = unit_vx * self.max_speed if mag > 0.05 else 0.0
            self.vy = unit_vy * self.max_speed if mag > 0.05 else 0.0
            return curr_wx, curr_wy, eff_vx_world, eff_vy_world

        # 1. Stuck check: if oscillating or stuck targeting the same spot, temporarily blacklist it
        self.pos_check_timer += 1
        if self.pos_check_timer >= 15:  # Every 0.3s
            self.pos_check_timer = 0
            self.pos_history.append((self.x, self.y))
            if len(self.pos_history) >= 4:
                p0 = self.pos_history[0]
                p_now = self.pos_history[-1]
                net_dist = math.hypot(p_now[0] - p0[0], p_now[1] - p0[1])
                if net_dist < 0.25:
                    if self._route_target is not None:
                        self._unreachable_targets[self._route_target] = self.frame_counter
                    self._route = []
                    self._route_target = None
                    self.pos_history.clear()
                    self.steer_vx = random.uniform(-0.8, 0.8)
                    self.steer_vy = random.uniform(-0.8, 0.8)

        # Clean expired unreachable targets (after 150 frames = 3.0s)
        if self._unreachable_targets:
            self._unreachable_targets = {
                tgt: f for tgt, f in self._unreachable_targets.items()
                if self.frame_counter - f < 150
            }

        # 2. Check if route needs replanning
        min_ghost_dist = float('inf')
        for g in ghosts.values():
            if not g.dead:
                gd = math.hypot(self.y - g.y, self.x - g.x)
                if gd < min_ghost_dist:
                    min_ghost_dist = gd

        ghost_emergency = not self.powered and min_ghost_dist < self.emergency_dist and self._route_age >= 6
        power_changed = self.powered != self._route_power_state
        target_eaten = False
        if self._route_target is not None and not self.powered:
            pellet_set = getattr(self.world, 'pellet_set', None)
            power_pellet_set = getattr(self.world, 'power_pellet_set', None)
            tx, ty = self._route_target[0], self._route_target[1]
            if pellet_set is not None and power_pellet_set is not None:
                has_p = (self._route_target in pellet_set) or any(abs(p[0] - tx) < 0.2 and abs(p[1] - ty) < 0.2 for p in pellet_set)
                has_pp = (self._route_target in power_pellet_set) or any(abs(p[0] - tx) < 0.2 and abs(p[1] - ty) < 0.2 for p in power_pellet_set)
                target_eaten = not (has_p or has_pp)
            else:
                has_p = (self._route_target in self.world.pellets) or any(abs(p[0] - tx) < 0.2 and abs(p[1] - ty) < 0.2 for p in self.world.pellets)
                has_pp = (self._route_target in self.world.power_pellets) or any(abs(p[0] - tx) < 0.2 and abs(p[1] - ty) < 0.2 for p in self.world.power_pellets)
                target_eaten = not (has_p or has_pp)

        path_exhausted = not self._route
        needs_replan = (path_exhausted or ghost_emergency or power_changed or target_eaten or self._route_age > self.replan_age)
        if needs_replan and not ghost_emergency and not path_exhausted and self._route_age < 10:
            needs_replan = False

        if needs_replan:
            target, path = self._pick_target(ghosts)
            self._route = path
            self._route_target = target
            self._route_power_state = self.powered
            self._route_age = 0

        # 3. Pop waypoints reached
        while self._route:
            d = math.hypot(self.y - self._route[0][0], self.x - self._route[0][1])
            if len(self._route) > 1:
                # Intermediate waypoint: pop when within 0.65 or overshot along heading
                wp_r, wp_c = self._route[0]
                dot = (self.x - wp_c) * self.vx + (self.y - wp_r) * self.vy
                if d < 0.65 or (dot > 0 and d < 0.85):
                    self._route.pop(0)
                else:
                    break
            else:
                # Final waypoint: DO NOT pop prematurely!
                # Only pop when very close (< 0.20) or eaten by referee.
                # Popping at 0.65 caused premature halts outside pellet pickup radius.
                if d < 0.20:
                    self._route.pop(0)
                break

        # 4. Desired heading towards current target waypoint
        if self._route:
            wp_r, wp_c = self._route[0]
            dr = wp_r - self.y
            dc = wp_c - self.x
            h_len = math.hypot(dr, dc)
            if h_len > 0.01:
                desired_vy = dr / h_len
                desired_vx = dc / h_len
            else:
                desired_vy = 0.0
                desired_vx = 0.0
        else:
            h_len = 0.0
            desired_vy = 0.0
            desired_vx = 0.0

        # 5. 16-ray context steering calculation
        num_rays = _NUM_STEERING_RAYS
        cur_speed_mag = math.hypot(self.steer_vx, self.steer_vy) + 1e-6
        cur_vx_norm = self.steer_vx / cur_speed_mag
        cur_vy_norm = self.steer_vy / cur_speed_mag
        ray_vx_arr = _STEERING_RAY_VX
        ray_vy_arr = _STEERING_RAY_VY

        # In doorways and tight corridors, scale ray lookahead to avoid false wall-hits
        # on lateral door jambs when steering towards nearby waypoints
        if self._route and h_len > 0.01:
            eff_check_dist = min(_STEERING_CHECK_DIST, max(0.9, h_len + 0.25))
        else:
            eff_check_dist = _STEERING_CHECK_DIST

        fracs = np.linspace(1 / _STEERING_N_STEPS, 1.0, _STEERING_N_STEPS)
        dx_grid = np.outer(ray_vx_arr, fracs) * eff_check_dist
        dy_grid = np.outer(ray_vy_arr, fracs) * eff_check_dist
        cc_grid = self.x + dx_grid
        cr_grid = self.y + dy_grid

        passable = self.world.batch_is_passable(cc_grid.flatten(), cr_grid.flatten(), self.radius).reshape((num_rays, _STEERING_N_STEPS))
        hit_mask = ~passable
        hit_indices = np.argmax(hit_mask, axis=1)
        has_hit = np.any(hit_mask, axis=1)
        hit_fracs = (hit_indices + 1) / _STEERING_N_STEPS
        hit_dists = (hit_indices + 1) * (eff_check_dist / _STEERING_N_STEPS)

        # Mask penalties for walls BEHIND the target waypoint:
        # When moving towards a doorway opening or corner pellet, walls behind the waypoint should not repel Pac-Man!
        if self._route and h_len < eff_check_dist:
            ray_align = ray_vx_arr * desired_vx + ray_vy_arr * desired_vy
            behind_target = (ray_align > 0.5) & (hit_dists > h_len + 0.12)
            effective_hits = has_hit & ~behind_target
        else:
            effective_hits = has_hit

        ray_penalties = np.where(effective_hits, 1000.0 / hit_fracs, 0.0)
        interests = ray_vx_arr * desired_vx + ray_vy_arr * desired_vy
        hysteresis = 0.35 * (ray_vx_arr * cur_vx_norm + ray_vy_arr * cur_vy_norm)

        if not self.powered and ghosts:
            ghost_penalties = np.zeros(_NUM_STEERING_RAYS)
            for g in ghosts.values():
                if not g.dead:
                    g_dx = g.x - self.x
                    g_dy = g.y - self.y
                    g_dist = math.hypot(g_dx, g_dy)
                    if g_dist < 4.5:
                        g_dir_x = g_dx / (g_dist + 1e-6)
                        g_dir_y = g_dy / (g_dist + 1e-6)
                        align = ray_vx_arr * g_dir_x + ray_vy_arr * g_dir_y
                        ghost_penalties += np.where(align > 0.1, (align ** 2) * (600.0 / max(g_dist, 0.4)), 0.0)
            scores = interests + hysteresis - ray_penalties - ghost_penalties
        elif self.powered and ghosts:
            ghost_attract = np.zeros(_NUM_STEERING_RAYS)
            for g in ghosts.values():
                if not g.dead:
                    g_dx = g.x - self.x
                    g_dy = g.y - self.y
                    g_dist = math.hypot(g_dx, g_dy)
                    if g_dist < 6.0:
                        g_dir_x = g_dx / (g_dist + 1e-6)
                        g_dir_y = g_dy / (g_dist + 1e-6)
                        align = ray_vx_arr * g_dir_x + ray_vy_arr * g_dir_y
                        ghost_attract += np.where(align > 0.2, align * 3.5, 0.0)
            scores = interests + hysteresis - ray_penalties + ghost_attract
        else:
            scores = interests + hysteresis - ray_penalties

        best_idx = np.argmax(scores)
        best_vx = float(ray_vx_arr[best_idx])
        best_vy = float(ray_vy_arr[best_idx])

        # Smooth steering momentum matching pacman.py with forward wall lookahead
        target_vx = best_vx
        target_vy = best_vy
        smooth_vx = self.steer_vx * 0.70 + target_vx * 0.30
        smooth_vy = self.steer_vy * 0.70 + target_vy * 0.30
        smooth_safe = True
        if self.world and hasattr(self.world, 'batch_is_passable'):
            smooth_mag = math.hypot(smooth_vx, smooth_vy)
            if smooth_mag > 1e-6:
                check_dist = min(eff_check_dist, smooth_mag * 1.2 + self.radius)
                n_steps_s = max(2, int(math.ceil(check_dist / 0.15)))
                s_vy_norm = smooth_vy / smooth_mag
                s_vx_norm = smooth_vx / smooth_mag
                fracs_s = np.linspace(1 / n_steps_s, 1.0, n_steps_s)
                cc_arr = self.x + s_vx_norm * check_dist * fracs_s
                cr_arr = self.y + s_vy_norm * check_dist * fracs_s
                passable = self.world.batch_is_passable(cc_arr, cr_arr, self.radius)
                smooth_safe = np.all(passable)
        if smooth_safe:
            self.steer_vx = smooth_vx
            self.steer_vy = smooth_vy
        else:
            self.steer_vx = target_vx
            self.steer_vy = target_vy

        steer_mag = math.hypot(self.steer_vx, self.steer_vy)
        if steer_mag > 0.02:
            unit_vx = self.steer_vx / steer_mag
            unit_vy = self.steer_vy / steer_mag
        else:
            unit_vx = target_vx
            unit_vy = target_vy

        # Physical speed calculation
        spd_mult = 1.15 if self.powered else 1.0
        cur_speed_metric = self.base_speed_metric * spd_mult
        v_grid_s = cur_speed_metric / self.scale

        # Convert starting position to world metrics for swept-path checking
        prev_wx = float((self.x - self.world.width / 2.0) * self.scale)
        prev_wy = float((self.y - self.world.height / 2.0) * self.scale)

        # Advance position along unit steering vector
        step_dist = v_grid_s * dt
        self.x += unit_vx * step_dist
        self.y += unit_vy * step_dist

        # Capsule wall collision resolution
        self.x, self.y = self.world.resolve_collision(self.x, self.y, self.radius, max_iters=3)

        # Update internal vx, vy
        self.vx = unit_vx * self.max_speed
        self.vy = unit_vy * self.max_speed

        # Convert to world metrics
        pac_wx = float((self.x - self.world.width / 2.0) * self.scale)
        pac_wy = float((self.y - self.world.height / 2.0) * self.scale)
        pac_vx_world = unit_vx * cur_speed_metric
        pac_vy_world = unit_vy * cur_speed_metric

        # Record continuous swept path for 1:1 pellet pickup and collision detection
        self.path_this_frame = [(prev_wx, prev_wy), (pac_wx, pac_wy)]

        return pac_wx, pac_wy, pac_vx_world, pac_vy_world


class PacmanGameEngine:
    """End-to-end continuous game engine running native MuJoCo physics and multi-agent AI."""

    def __init__(
        self,
        stage: int = 1,
        seed: Optional[int] = None,
        scale: float = 1.5,
        policy_ckpt: str = "checkpoints/svanm2_flat/svanm2_flat.pt",
        ghost_ckpt: int = -1,
        enable_rl: bool = True,
        world: Any = None,
        grid: Any = None,
        player_start_grid: Tuple[float, float] = None,
        ghost_starts_grid: List[Tuple[float, float]] = None,
    ):
        self.stage = stage
        self.scale = scale
        self.seed = seed

        # 1. Build procedural continuous arena & compile MuJoCo model
        self.arena = ContinuousArena(
            stage=stage,
            scale=scale,
            seed=seed,
            world=world,
            grid=grid,
            player_start_grid=player_start_grid,
            ghost_starts_grid=ghost_starts_grid,
        )
        self.model = self.arena.model
        self.data = self.arena.data

        # 2. Resolve Pac-Man root joint and body
        self.pac_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pacman_root")
        assert self.pac_jnt_id != -1, "pacman_root joint not found in model"
        self.pac_qpos_adr = self.model.jnt_qposadr[self.pac_jnt_id]
        self.pac_dof_adr = self.model.jnt_dofadr[self.pac_jnt_id]

        # 3. Load SvanM2 policy and create quadruped runners
        self.policy = SvanM2Policy(policy_ckpt)
        self.runners: List[SvanM2Runner] = []
        for gid in range(self.arena.n_ghosts):
            prefix = f"ghost_{gid}/"
            runner = SvanM2Runner(self.model, self.data, gid, prefix, self.policy)
            gx, gy = self.arena.ghost_starts[gid]
            runner.reset_pose(gx, gy, z=0.34)
            self.runners.append(runner)

        # 4. Initialize Multi-Agent Hive Coordinator with authentic MAPPO RL
        self.hive = HiveCoordinator(
            self.arena,
            self.runners,
            checkpoint=ghost_ckpt,
            enable_rl=enable_rl,
        )

        # 5. Initialize Continuous Pac-Man agent (Player from pacmanbot with continuous physics)
        pac_gx, pac_gy = self.arena.to_grid(self.arena.pacman_start[0], self.arena.pacman_start[1])

        # Calibrated authentic speed ratios matching pacmanbot:
        # SvanM2 quadrupeds: patrol trot = 0.80 m/s, hunt sprint = 0.95 m/s, evade = 0.52 m/s.
        # In pacmanbot/pacman.py, authentic ratio is ~1.5x - 2.0x (Pac-Man = 1.0, Ghost = 0.50).
        # Setting Pac-Man base speed to 1.40 m/s achieves authentic ratios:
        # 1.47x vs hunting sprint, 1.75x vs patrol trot, and 2.69x vs evading ghosts.
        v_pac_base_metric = 1.40  # m/s

        self.pacman = ContinuousPlayer(
            grid=self.arena.grid,
            pos=(float(pac_gy), float(pac_gx)),  # (row, col)
            world=self.arena.world,
            scale=self.scale,
            target_speed_metric=v_pac_base_metric,
        )

        self.pacman_yaw = 0.0
        self.prev_pac_wx = float(self.arena.pacman_start[0])
        self.prev_pac_wy = float(self.arena.pacman_start[1])

        # Set initial Pac-Man position in MuJoCo data (z = 0.18 ground height)
        self.data.qpos[self.pac_qpos_adr : self.pac_qpos_adr + 3] = [
            self.arena.pacman_start[0],
            self.arena.pacman_start[1],
            0.18,
        ]
        self.data.qpos[self.pac_qpos_adr + 3 : self.pac_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

        # Game state tracking
        self.score = 0
        self.game_over = False
        self.outcome = "PLAYING"  # "PLAYING", "PACMAN_WIN", "GHOSTS_WIN", "TIMEOUT"
        self.frame = 0
        self.dt = 0.02  # 50 Hz policy evaluation rate (10 physics sub-steps at 500 Hz)
        # Power pellet duration derived from straight-line clearance distance: T = D / S
        # D: clearance distance in grid cells that Pac-Man can travel in a straight line at max speed
        self.power_clearance_dist_cells = float(os.environ.get("POWER_CLEARANCE_DIST_CELLS", "7.5"))
        pac_max_speed_metric = getattr(self.pacman, 'base_speed_metric', 1.40) * 1.15
        pac_speed_cells_per_step = (pac_max_speed_metric / self.scale) * self.dt
        self.power_duration_steps = max(1, int(round(self.power_clearance_dist_cells / max(pac_speed_cells_per_step, 1e-4))))
        self.power_timer_steps = 0  # Remaining policy steps of powered state

        # Propagate power duration steps to hive coordinator and ghost instances
        self.hive.power_duration_steps = self.power_duration_steps
        for gh in self.hive.ghosts.values():
            gh.power_duration_steps = self.power_duration_steps
            gh.pacman_power_max_timer = float(self.power_duration_steps)

        # Track active pellet geom IDs
        self.active_pellets = set(self.arena.pellets.keys())
        self.active_power = set(self.arena.power_pellets.keys())
        self.power_pellets_defused_total = 0
        self.power_pellets_eaten_total = 0

        # Proximity thresholds matching pacmanbot + capsule wall clearance:
        # eat_radius = 0.60 grid units -> 0.60 * scale meters (0.90m in 1.5m scale)
        # Prevents corner pellets from ever getting locked behind capsule boundary margins.
        self.pellet_pickup_dist = 0.60 * self.scale
        # Ghost defusal radius matching ghost.py (self.radius + 0.5 = 0.40 + 0.50 = 0.90) * scale = 1.35m
        self.power_defuse_dist = 0.90 * self.scale
        # Agent contact distance matching pacmanbot: (pacman_radius + ghost_radius + margin) * scale
        self.collision_dist = (self.pacman.radius + 0.40 + 0.15) * self.scale

        # Pre-resolve geom IDs to eliminate repeated mj_name2id string lookups in 50 Hz loop
        self._pac_geom_ids = [
            gid for pg in ["pacman_chassis", "pacman_dome", "pacman_hub_l", "pacman_hub_r"]
            if (gid := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, pg)) != -1
        ]
        self._ghost_geom_ids = [
            (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_{gid}/torso_0"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_{gid}/torso_1"),
            )
            for gid in range(self.arena.n_ghosts)
        ]
        self._pellet_geom_ids = {
            p_idx: gid
            for p_idx in self.arena.pellets
            if (gid := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"pellet_{p_idx}")) != -1
        }
        self._power_geom_ids = {
            pp_idx: gid
            for pp_idx in self.arena.power_pellets
            if (gid := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"power_{pp_idx}")) != -1
        }
        self._defused_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "defused_mat")

    @property
    def pacman_position(self) -> Tuple[float, float, float]:
        pos = self.data.qpos[self.pac_qpos_adr : self.pac_qpos_adr + 3]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    @property
    def is_pacman_powered(self) -> bool:
        return self.power_timer_steps > 0

    def step_simulation(self):
        """Advances the simulation by 1 policy step (0.020s = 50 Hz) and 10 PD sub-steps."""
        if self.game_over:
            return

        self.frame += 1

        # 1. Update Pac-Man power state countdown
        if self.power_timer_steps > 0:
            self.power_timer_steps -= 1
            self.pacman.power_timer = self.power_timer_steps
            if self.power_timer_steps == 0:
                self.pacman.powered = False
                self.pacman.power_timer = 0

        # Visual update for Pac-Man robot body (yellow gold vs electric blue surge)
        if self.is_pacman_powered:
            pac_col = [0.0, 0.47, 1.0, 1.0] if (self.power_timer_steps >= 100 or (self.frame // 8) % 2 == 0) else [0.90, 0.95, 1.0, 1.0]
        else:
            pac_col = [0.98, 0.78, 0.02, 1.0]
        for gid in self._pac_geom_ids:
            self.model.geom_rgba[gid] = pac_col

        # 2. Update Pac-Man autonomous navigation (ContinuousPlayer PRM + 16-ray context steering)
        pac_wx, pac_wy, pac_vx_world, pac_vy_world = self.pacman.step_continuous(
            dt=0.02,
            ghosts=self.hive.ghosts,
            powered=self.is_pacman_powered,
        )

        # Smooth heading orientation towards movement direction (fluid AMR turning without visual twitch)
        v_mag = math.hypot(pac_vx_world, pac_vy_world)
        if v_mag > 0.03:
            target_yaw = math.atan2(pac_vy_world, pac_vx_world)
            yaw_diff = math.atan2(math.sin(target_yaw - self.pacman_yaw), math.cos(target_yaw - self.pacman_yaw))
            self.pacman_yaw += float(np.clip(yaw_diff * 0.25, -0.15, 0.15))

        qw = math.cos(self.pacman_yaw / 2.0)
        qz = math.sin(self.pacman_yaw / 2.0)
        self.data.qpos[self.pac_qpos_adr : self.pac_qpos_adr + 3] = [pac_wx, pac_wy, 0.18]
        self.data.qpos[self.pac_qpos_adr + 3 : self.pac_qpos_adr + 7] = [qw, 0.0, 0.0, qz]
        self.data.qvel[self.pac_dof_adr : self.pac_dof_adr + 2] = [pac_vx_world, pac_vy_world]
        self.data.qvel[self.pac_dof_adr + 2] = 0.0

        # 3. Update ghost body colors (SvanM2 torso geoms always maintain distinct ghost colors)
        for gid in range(self.arena.n_ghosts):
            t0_id, t1_id = self._ghost_geom_ids[gid]
            if self.hive.ghosts[gid].dead:
                g_col = [0.0, 0.0, 0.0, 0.0]
            else:
                col = GHOST_COLORS[gid % len(GHOST_COLORS)]
                g_col = [col[0], col[1], col[2], 1.0]

            for tid in (t0_id, t1_id):
                if tid != -1:
                    self.model.geom_rgba[tid] = g_col

        # 4. Step Hive Coordinator (LiDAR sweeps, radio mesh, CBBA, context steering)
        self.hive.step((pac_wx, pac_wy), self.is_pacman_powered)

        # Update ghost 3D robot root poses and heading orientation from exact ghost autonomy state
        ghost_substep_poses = []
        for gid, runner in enumerate(self.runners):
            g = self.hive.ghosts[gid]
            if not g.dead:
                gw_x, gw_y = self.arena.to_world(g.x, g.y)
                if runner.filtered_target_yaw is not None:
                    gyaw = runner.filtered_target_yaw
                else:
                    g_vmag = math.hypot(g.vx, g.vy)
                    if g_vmag > 1e-4:
                        tyaw = math.atan2(g.vy, g.vx)
                        cur_yaw = getattr(runner, 'target_yaw', tyaw)
                        ydiff = math.atan2(math.sin(tyaw - cur_yaw), math.cos(tyaw - cur_yaw))
                        runner.target_yaw = cur_yaw + float(np.clip(ydiff * 0.35, -0.25, 0.25))
                    gyaw = getattr(runner, 'target_yaw', 0.0)

                gqw = math.cos(gyaw / 2.0)
                gqz = math.sin(gyaw / 2.0)

                # Project base-frame commanded velocity into world coordinates:
                vx_w = math.cos(gyaw) * runner.cmd_base[0] - math.sin(gyaw) * runner.cmd_base[1]
                vy_w = math.sin(gyaw) * runner.cmd_base[0] + math.cos(gyaw) * runner.cmd_base[1]
                wz_w = float(runner.cmd_base[2])

                self.data.qpos[runner.root_qpos_adr : runner.root_qpos_adr + 3] = [gw_x, gw_y, 0.34]
                self.data.qpos[runner.root_qpos_adr + 3 : runner.root_qpos_adr + 7] = [gqw, 0.0, 0.0, gqz]
                self.data.qvel[runner.root_dof_adr : runner.root_dof_adr + 3] = [vx_w, vy_w, 0.0]
                self.data.qvel[runner.root_dof_adr + 3 : runner.root_dof_adr + 6] = [0.0, 0.0, wz_w]
                ghost_substep_poses.append((runner, gw_x, gw_y, gqw, gqz, vx_w, vy_w, wz_w))

        # 5. Step Quadruped Locomotion Policies (50 Hz)
        for runner in self.runners:
            runner.recover_if_fallen()
            runner.step_policy()

        # 6. Execute 10 Physics Sub-Steps at 500 Hz (dt = 0.002s)
        for _ in range(10):
            self.data.qpos[self.pac_qpos_adr : self.pac_qpos_adr + 3] = [pac_wx, pac_wy, 0.18]
            self.data.qpos[self.pac_qpos_adr + 3 : self.pac_qpos_adr + 7] = [qw, 0.0, 0.0, qz]
            self.data.qvel[self.pac_dof_adr : self.pac_dof_adr + 2] = [pac_vx_world, pac_vy_world]
            self.data.qvel[self.pac_dof_adr + 2] = 0.0

            for runner, gw_x, gw_y, gqw, gqz, vx_w, vy_w, wz_w in ghost_substep_poses:
                self.data.qpos[runner.root_qpos_adr : runner.root_qpos_adr + 3] = [gw_x, gw_y, 0.34]
                self.data.qpos[runner.root_qpos_adr + 3 : runner.root_qpos_adr + 7] = [gqw, 0.0, 0.0, gqz]
                self.data.qvel[runner.root_dof_adr : runner.root_dof_adr + 3] = [vx_w, vy_w, 0.0]
                self.data.qvel[runner.root_dof_adr + 3 : runner.root_dof_adr + 6] = [0.0, 0.0, wz_w]

            # Compute SvanM2 parallel mechanism belt transmission torques
            for runner in self.runners:
                runner.step_pd()

            mujoco.mj_step(self.model, self.data)

        # 7. Referee checks: Pellet pickups, Power pellet defusals, Agent collisions
        self._check_referee_events()

    def _check_referee_events(self):
        """Evaluates pellet consumption, power defusal, and agent collisions."""
        pac_x, pac_y, _ = self.pacman_position

        # Continuous point-to-segment swept check (matching pacman.py's path_this_frame)
        p_path = getattr(self.pacman, 'path_this_frame', None)

        def _dist_to_pac(tx: float, ty: float) -> float:
            if p_path and len(p_path) == 2:
                (x1, y1), (x2, y2) = p_path
                dx, dy = x2 - x1, y2 - y1
                l2 = dx * dx + dy * dy
                if l2 > 1e-8:
                    t = max(0.0, min(1.0, ((tx - x1) * dx + (ty - y1) * dy) / l2))
                    return math.hypot(tx - (x1 + t * dx), ty - (y1 + t * dy))
            return math.hypot(pac_x - tx, pac_y - ty)

        # A. Pac-Man vs Regular Pellets
        eaten_pellets = []
        for p_idx in list(self.active_pellets):
            px, py = self.arena.pellets[p_idx]
            if _dist_to_pac(px, py) <= self.pellet_pickup_dist:
                eaten_pellets.append(p_idx)
                self.score += 10
                self.pacman.score = self.score
                # Move geom underground to visually hide it
                gid = self._pellet_geom_ids.get(p_idx, -1)
                if gid != -1:
                    self.model.geom_pos[gid][2] = -10.0
                # Synchronize with pacmanbot world model
                pt = self.arena.pellet_grid_pts.get(p_idx)
                if pt and pt in self.arena.world.pellets:
                    self.arena.world.pellets.remove(pt)
                    self.arena.world._update_pellet_arrays()

        for p_idx in eaten_pellets:
            self.active_pellets.remove(p_idx)

        # B. Pac-Man vs Power Pellets
        eaten_power = []
        for pp_idx in list(self.active_power):
            ppx, ppy = self.arena.power_pellets[pp_idx]
            if _dist_to_pac(ppx, ppy) <= self.pellet_pickup_dist:
                eaten_power.append(pp_idx)
                self.power_pellets_eaten_total += 1
                self.score += 50
                self.pacman.score = self.score
                # Activate power state immediately
                self.power_timer_steps = self.power_duration_steps
                self.pacman.powered = True
                self.pacman.power_timer = self.power_duration_steps
                # Move geom underground
                gid = self._power_geom_ids.get(pp_idx, -1)
                if gid != -1:
                    self.model.geom_pos[gid][2] = -10.0
                # Synchronize with pacmanbot world model
                pt = self.arena.power_grid_pts.get(pp_idx)
                if pt and pt in self.arena.world.power_pellets:
                    self.arena.world.power_pellets.remove(pt)
                    self.arena.world._update_pellet_arrays()

        for pp_idx in eaten_power:
            self.active_power.remove(pp_idx)

        # C. Ghost vs Power Pellets (CONVERT rule / Defusal)
        # When a ghost reaches an un-eaten power pellet, it defuses it into a regular 10-pt pellet
        defused_power = []
        for pp_idx in list(self.active_power):
            ppx, ppy = self.arena.power_pellets[pp_idx]
            pt = self.arena.power_grid_pts.get(pp_idx)
            # Bi-directional check: did ghost world autonomy defuse it already?
            if pt and pt not in self.arena.world.power_pellets:
                defused_power.append((pp_idx, 0))
                continue
            for gid, runner in enumerate(self.runners):
                if self.hive.ghosts[gid].dead:
                    continue
                gx, gy, _ = runner.position
                if math.hypot(gx - ppx, gy - ppy) <= self.power_defuse_dist:
                    defused_power.append((pp_idx, gid))
                    break

        for pp_idx, defusing_gid in defused_power:
            if pp_idx not in self.active_power:
                continue
            self.active_power.remove(pp_idx)
            self.power_pellets_defused_total += 1
            ppx, ppy = self.arena.power_pellets[pp_idx]

            # Change material/size in MuJoCo to defused regular pellet
            geom_id = self._power_geom_ids.get(pp_idx, -1)
            if geom_id != -1:
                if self._defused_mat_id != -1:
                    self.model.geom_matid[geom_id] = self._defused_mat_id
                self.model.geom_size[geom_id][0] = 0.09  # Shrink to regular pellet size
                self.model.geom_rgba[geom_id] = [0.35, 0.85, 1.0, 0.9]

            # Synchronize with pacmanbot world model
            pt = self.arena.power_grid_pts.get(pp_idx)
            if pt:
                if pt in self.arena.world.power_pellets:
                    self.arena.world.power_pellets.remove(pt)
                if pt not in self.arena.world.pellets:
                    self.arena.world.pellets.append(pt)
                self.arena.world._update_pellet_arrays()

                # Update ghost swarm local knowledge across all ghosts
                for gh in self.hive.ghosts.values():
                    gh.known_power_pellets.discard(pt)
                    gh.known_pellets.add(pt)
                gh = self.hive.ghosts[defusing_gid]
                gh.power_pellets_converted_this_frame += 1
                gh._broadcast([("power_eaten", pt), ("pellet", pt)], self.hive.ghosts)

            # Add to active pellets as a regular pellet
            defused_key = f"defused_{pp_idx}"
            self.arena.pellets[defused_key] = (ppx, ppy)
            self.arena.pellet_grid_pts[defused_key] = pt
            self.active_pellets.add(defused_key)

        # D. Pac-Man vs Ghost Collisions (Exact Swept-to-Swept Continuous Collision Detection)
        p_path = getattr(self.pacman, 'path_this_frame', None) or [(pac_x, pac_y)]
        n_p = len(p_path)

        for gid, runner in enumerate(self.runners):
            g = self.hive.ghosts[gid]
            if g.dead:
                continue

            g_path_grid = getattr(g, 'path_this_frame', None) or [(g.x, g.y)]
            g_path = [self.arena.to_world(pt[0], pt[1]) for pt in g_path_grid]
            if not g_path:
                gx, gy, _ = runner.position
                g_path = [(gx, gy)]
            n_g = len(g_path)

            # Continuous swept segment interpolation matching pacmanbot/pacman.py:L1102-L1130
            max_segs = max(1, n_p - 1, n_g - 1)
            samples = max_segs * 4 + 1
            collided = False
            for step in range(samples):
                t = step / (samples - 1) if samples > 1 else 0.0
                if n_p == 1:
                    px, py = p_path[0]
                else:
                    fp = t * (n_p - 1)
                    ip = min(int(fp), n_p - 2)
                    rem_p = fp - ip
                    px = p_path[ip][0] * (1.0 - rem_p) + p_path[ip + 1][0] * rem_p
                    py = p_path[ip][1] * (1.0 - rem_p) + p_path[ip + 1][1] * rem_p
                if n_g == 1:
                    gx, gy = g_path[0]
                else:
                    fg = t * (n_g - 1)
                    ig = min(int(fg), n_g - 2)
                    rem_g = fg - ig
                    gx = g_path[ig][0] * (1.0 - rem_g) + g_path[ig + 1][0] * rem_g
                    gy = g_path[ig][1] * (1.0 - rem_g) + g_path[ig + 1][1] * rem_g
                if math.hypot(gx - px, gy - py) <= self.collision_dist:
                    collided = True
                    break

            if not collided:
                # Instantaneous robot center swept check
                gx, gy, _ = runner.position
                if _dist_to_pac(gx, gy) <= self.collision_dist:
                    collided = True

            if collided:
                if self.is_pacman_powered:
                    # Pac-Man eats Ghost!
                    self.hive.eliminate_ghost(gid)
                    self.score += 200
                    self.pacman.score = self.score
                else:
                    # Ghost captures Pac-Man!
                    self.game_over = True
                    self.outcome = "GHOSTS_WIN"
                    return

        # Reset per-frame defusal counters on all ghosts
        for g in self.hive.ghosts.values():
            g.power_pellets_converted_this_frame = 0

        # Check Win Conditions
        if len(self.active_pellets) == 0 and len(self.active_power) == 0:
            self.game_over = True
            self.outcome = "PACMAN_WIN"
        elif self.hive.alive_ghost_count == 0:
            self.game_over = True
            self.outcome = "PACMAN_WIN"
