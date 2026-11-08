#!/usr/bin/env python3
"""
compare_simulations.py — Headless Comparison Benchmark: Native MuJoCo vs pacman.py.

Runs both simulations on the EXACT same procedural maze, pellets, and spawn points,
and rigorously compares:
  1. Game Outcome (PACMAN_WIN vs GHOSTS_WIN vs TIMEOUT) and Concurrence Rate
  2. Survival Duration and Speed of Play (simulated seconds)
  3. Final Scores and Regular/Power Pellet Clearance Percentages
  4. Power Pellet Defusal Counts (CONVERT rule executed by ghost swarm)
  5. Spatial Trajectory Divergence:
     • Pointwise Mean & Peak Euclidean separation distance d(t)
     • Discrete Fréchet Distance (curve shape divergence)
     • Time/Step of first trajectory branch divergence
  6. Physical Reality Gap & Kinematics:
     • Quadruped locomotion vs Kinematic stepping
     • Average realized velocity and speed ratios
  7. Detailed event timestamps (pellet milestones, defusals, fatal contacts)

Features:
  • 100% Headless execution (no X11 / display server required)
  • Single-run detailed analysis or multi-run statistical benchmark
  • Rich ASCII comparison summary table
  • High-resolution multi-panel diagnostic plot (--plot output.png)
  • Machine-readable JSON export (--output output.json)

Usage:
  python scripts/compare_simulations.py --stage 1 --seed 42
  python scripts/compare_simulations.py --stage 2 --num_runs 5
  python scripts/compare_simulations.py --stage 1 --seed 42 --plot comparison.png --output report.json
"""

import os
import sys
import math
import time
import copy
import json
import glob
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Ensure headless execution before importing GUI-dependent libraries
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# Add repository roots to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"

for p in [str(_PACMANBOT_DIR), str(_PACMAN_SIM_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import pygame

# pacmanbot components
from curriculum import STAGES
from pacman import (
    generate_map,
    Player,
    WALL,
    EMPTY,
    PELLET,
    POWER,
    GHOST_COLORS,
    _NUM_STEERING_RAYS,
)
from ghost import Ghost, GHOST_SPEED
from net import GhostActor, speed_idx_to_mult
from obs import (
    build_spatial,
    build_vector,
    build_valid_mask,
    actions_to_tasks,
    MAX_H,
    MAX_W,
)
from allocator import generate_tasks as _heuristic_tasks, Task, TaskType, ORIGIN_RL_NOVEL
from cbba import _deduplicate_tasks
from pathfinder import dijkstra_multi

# pacman_sim MuJoCo components
from game_engine import PacmanGameEngine


# =============================================================================
# Helper: Discrete Fréchet Distance for Trajectory Shape Divergence
# =============================================================================
def discrete_frechet_distance(p: np.ndarray, q: np.ndarray, max_points: int = 350) -> float:
    """
    Computes discrete Fréchet distance between two 2D polygonal curves p and q.
    Measures geometric trajectory divergence independent of pacing.
    """
    if len(p) == 0 or len(q) == 0:
        return 0.0

    # Downsample if trajectories are very long for fast sub-second computation
    if len(p) > max_points:
        idx_p = np.linspace(0, len(p) - 1, max_points).astype(int)
        p = p[idx_p]
    if len(q) > max_points:
        idx_q = np.linspace(0, len(q) - 1, max_points).astype(int)
        q = q[idx_q]

    n_p = len(p)
    n_q = len(q)
    ca = np.full((n_p, n_q), -1.0, dtype=np.float64)

    ca[0, 0] = np.linalg.norm(p[0] - q[0])
    for j in range(1, n_q):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(p[0] - q[j]))
    for i in range(1, n_p):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(p[i] - q[0]))

    for i in range(1, n_p):
        for j in range(1, n_q):
            d = np.linalg.norm(p[i] - q[j])
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), d)

    return float(ca[n_p - 1, n_q - 1])


# =============================================================================
# Shared Deterministic Map & Spawn Generator
# =============================================================================
def setup_shared_map(stage_idx: int, seed: int):
    """
    Generates a deterministic continuous maze and spawn points matching pacman.py
    and returns independent deep copies for both simulations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg_idx = max(0, min(len(STAGES) - 1, stage_idx - 1))
    stage_cfg = STAGES[cfg_idx]
    raw_h = stage_cfg.rows
    raw_w = stage_cfg.cols
    n_ghosts = stage_cfg.n_ghosts
    n_power = stage_cfg.n_power

    # 1. Generate map using pacmanbot
    grid, player_start, world = generate_map(
        world_height=float(raw_h),
        world_width=float(raw_w),
        n_power=n_power,
        random_spawn=False,
        obs_resolution=1.0,
    )

    # 2. Compute ghost spawn points matching pacman.py:new_game()
    open_cells_arr = None
    if hasattr(world, "prm_nodes") and world.prm_nodes:
        open_cells_arr = np.array(world.prm_nodes)
    if open_cells_arr is None or len(open_cells_arr) < n_ghosts:
        safe = getattr(world, "safe_area", [])
        if safe:
            sampled = random.sample(safe, min(len(safe), max(n_ghosts * 3, 20)))
            open_cells_arr = np.array([(pt[1], pt[0]) for pt in sampled])
        else:
            open_cells_arr = np.array([[float(player_start[0]), float(player_start[1])]])

    pac_yx = np.array([float(player_start[0]), float(player_start[1])])
    d_pac = np.sum(np.square(open_cells_arr - pac_yx), axis=1)
    avail = np.ones(len(open_cells_arr), dtype=bool)

    first_idx = int(np.argmax(d_pac))
    ghost_starts = [tuple(open_cells_arr[first_idx])]
    avail[first_idx] = False
    min_dist_to_ghosts = np.full(len(open_cells_arr), np.inf)

    for _ in range(n_ghosts - 1):
        last_placed = np.array(ghost_starts[-1])
        dist_to_last = np.sum(np.square(open_cells_arr - last_placed), axis=1)
        min_dist_to_ghosts = np.minimum(min_dist_to_ghosts, dist_to_last)
        scores = np.minimum(d_pac, min_dist_to_ghosts)
        scores[~avail] = -1
        best_idx = int(np.argmax(scores))
        ghost_starts.append(tuple(open_cells_arr[best_idx]))
        avail[best_idx] = False

    # 3. Create independent deep copies for MuJoCo and pacman.py
    world_mj = copy.deepcopy(world)
    grid_mj = np.copy(grid)
    world_py = copy.deepcopy(world)
    grid_py = np.copy(grid)

    return {
        "stage_cfg": stage_cfg,
        "player_start": player_start,
        "ghost_starts": ghost_starts,
        "world_mj": world_mj,
        "grid_mj": grid_mj,
        "world_py": world_py,
        "grid_py": grid_py,
        "total_pellets": len(world.pellets),
        "total_power": len(world.power_pellets),
    }


# =============================================================================
# Headless Runner 1: pacman.py (2D Kinematic Game Engine)
# =============================================================================
class HeadlessPacmanPyRunner:
    """Runs pacman.py's exact multi-agent game loop 100% headlessly."""

    def __init__(
        self,
        grid: np.ndarray,
        player_start: Tuple[float, float],
        ghost_starts: List[Tuple[float, float]],
        world,
        enable_rl: bool = True,
        ghost_ckpt: int = -1,
        dt_step: float = 0.020,  # 50 Hz decision/physics frame
        scale: float = 1.5,
        speed_mode: str = "calibrated",  # "calibrated" (physical m/s) or "raw" (discrete steps)
    ):
        self.grid = grid
        self.player_start = player_start
        self.ghost_starts = ghost_starts
        self.world = world
        self.enable_rl = enable_rl
        self.dt_step = dt_step
        self.scale = scale
        self.speed_mode = speed_mode

        if self.speed_mode == "calibrated":
            from game_engine import ContinuousPlayer
            self.player = ContinuousPlayer(
                grid=self.grid,
                pos=self.player_start,
                world=self.world,
                scale=self.scale,
                target_speed_metric=1.40,
            )
        else:
            self.player = Player(self.grid, self.player_start, self.world)

        self.ghosts = {
            i: Ghost(i, self.grid, pos, GHOST_COLORS[i % len(GHOST_COLORS)], self.player_start, self.world)
            for i, pos in enumerate(ghost_starts)
        }

        # In calibrated mode, match the continuous metric speed of MuJoCo quadrupeds
        if self.speed_mode == "calibrated":
            for g in self.ghosts.values():
                g.max_speed = (0.80 / self.scale) * self.dt_step

        # Power pellet duration derived from straight-line clearance distance: T = D / S
        self.power_clearance_dist_cells = float(os.environ.get("POWER_CLEARANCE_DIST_CELLS", "7.5"))
        if self.speed_mode == "calibrated":
            p_spd_step = (1.40 * 1.15 / self.scale) * self.dt_step
        else:
            p_spd_step = 1.0 * 1.15
        self.power_duration_steps = max(1, int(round(self.power_clearance_dist_cells / max(p_spd_step, 1e-4))))

        for g in self.ghosts.values():
            g.power_duration_steps = self.power_duration_steps
            g.pacman_power_max_timer = float(self.power_duration_steps)

        # Initialize MAPPO RL Actor if enabled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rl_actor = None
        if self.enable_rl:
            self._load_rl_model(ghost_ckpt)

        for g in self.ghosts.values():
            g.rl_mode = (self.rl_actor is not None)
            g.cbba_agent.rl_mode = (self.rl_actor is not None)

        self.recent_nom = {i: np.zeros((MAX_H, MAX_W), dtype=np.float32) for i in self.ghosts}
        self.frame_counter = 0
        self.sim_time = 0.0
        self.game_over = False
        self.outcome = "PLAYING"

        self.initial_pellets_count = len(self.world.pellets)
        self.initial_power_count = len(self.world.power_pellets)

        # Logging traces
        self.pacman_traj: List[Tuple[float, float, float]] = []  # (t, x, y)
        self.ghost_trajs: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in self.ghosts}
        self.pacman_speeds: List[float] = []
        self.ghost_speeds: List[float] = []
        self.pellet_history: List[Tuple[float, int]] = []
        self.score_history: List[Tuple[float, int]] = []
        self.events: List[Dict[str, Any]] = []

        self.power_pellets_defused_total = 0
        self.ghosts_eaten_total = 0
        self.power_pellets_eaten_total = 0

    def _load_rl_model(self, checkpoint: int):
        try:
            ckpts = sorted(
                glob.glob(str(_PACMANBOT_DIR / "checkpoints" / "ckpt_*.pt")),
                key=lambda f: int(f.split("ckpt_")[-1].split(".pt")[0]) if "ckpt_" in f else 0,
            )
            if not ckpts:
                self.enable_rl = False
                return

            if checkpoint != -1:
                target = str(_PACMANBOT_DIR / "checkpoints" / f"ckpt_{checkpoint}.pt")
                chosen = target if Path(target).exists() else ckpts[-1]
            else:
                chosen = ckpts[-1]

            sd = torch.load(chosen, map_location=self.device, weights_only=False)
            actor = GhostActor(n_cand=3).to(self.device)
            actor_sd = sd.get("actor", sd)
            actor.load_state_dict(actor_sd, strict=False)
            actor.eval()
            self.rl_actor = actor

            pred_sd = sd.get("predictor")
            if pred_sd:
                for g in self.ghosts.values():
                    if hasattr(g, "belief_map") and hasattr(g.belief_map, "predictor"):
                        try:
                            g.belief_map.predictor.load_state_dict(pred_sd)
                        except Exception:
                            pass
        except Exception as e:
            self.enable_rl = False
            self.rl_actor = None

    def step(self):
        if self.game_over:
            return

        self.frame_counter += 1
        self.sim_time += self.dt_step

        alive = [gid for gid, g in self.ghosts.items() if not g.dead]

        # 1. High-level MAPPO RL inference and CBBA consensus (every 6 frames = 0.12s)
        if self.rl_actor is not None and (self.frame_counter % 6 == 0) and alive:
            R = min(MAX_H, len(self.grid))
            C = min(MAX_W, len(self.grid[0]))

            sp, ve, vm = [], [], []
            for gid in alive:
                g = self.ghosts[gid]
                s = build_spatial(g, self.recent_nom[gid], R, C, obs_resolution=1.0)
                v = build_vector(g)
                m = build_valid_mask(g, R, C, obs_resolution=1.0, spatial_walls=s[0])

                def _pad(arr):
                    h, w = arr.shape[-2], arr.shape[-1]
                    if h == MAX_H and w == MAX_W:
                        return arr
                    out = np.zeros(arr.shape[:-2] + (MAX_H, MAX_W), dtype=arr.dtype)
                    out[..., :h, :w] = arr
                    return out

                sp.append(_pad(s))
                ve.append(v)
                vm.append(_pad(m))

            t_sp = torch.tensor(np.stack(sp), device=self.device, dtype=torch.float32)
            t_ve = torch.tensor(np.stack(ve), device=self.device, dtype=torch.float32)
            t_vm = torch.tensor(np.stack(vm), device=self.device, dtype=torch.bool)

            with torch.inference_mode():
                out_act = self.rl_actor(t_sp, t_ve, t_vm, K=3)
                sel_idx, _, scores, _, _, speed_idx, _, _ = out_act

            idx_np = sel_idx.cpu().numpy()
            sc_np = scores.cpu().numpy()
            spd_np = speed_idx.cpu().numpy()

            for i, gid in enumerate(alive):
                g = self.ghosts[gid]
                g.current_speed_mult = speed_idx_to_mult(spd_np[i])
                g.current_rl_dir = None
                g.rl_hijack = False
                g.rl_mode = True
                indices = [(int(x // MAX_W), int(x % MAX_W)) for x in idx_np[i] if int(x // MAX_W) < R and int(x % MAX_W) < C]
                self.recent_nom[gid] *= 0.8
                for r_m, c_m in indices:
                    if 0 <= r_m < R and 0 <= c_m < C:
                        self.recent_nom[gid][r_m, c_m] = 1.0

            pool_tasks = []
            for i, gid in enumerate(alive):
                g = self.ghosts[gid]
                indices = [(int(x // MAX_W), int(x % MAX_W)) for x in idx_np[i] if int(x // MAX_W) < R and int(x % MAX_W) < C]
                rl_tasks = actions_to_tasks(g, sc_np[i], indices, self.frame_counter, target_speed=g.current_speed_mult)
                if rl_tasks:
                    pool_tasks.extend(rl_tasks)

            deduped_pool = _deduplicate_tasks(pool_tasks, threshold=1.5)
            all_targets = [t.target_pos for t in deduped_pool]
            for gid in alive:
                g = self.ghosts[gid]
                d_rl = dijkstra_multi(g.world, (g.y, g.x), all_targets) if all_targets else {}
                g.cbba_agent._last_auction = self.frame_counter + 6
                g.cbba_agent._phase1(g, deduped_pool, d_rl)

            if len(alive) > 1:
                for round_idx in range(2):
                    payloads = {gid: self.ghosts[gid].cbba_agent.get_consensus_payload() for gid in alive}
                    for gid_i in alive:
                        agent_i = self.ghosts[gid_i].cbba_agent
                        for gid_j in alive:
                            if gid_i != gid_j:
                                p_j = payloads[gid_j]
                                agent_i.receive_consensus(gid_j, p_j["y"], p_j["z"], p_j["s"], self.frame_counter, p_j.get("meta"))
                    if round_idx < 1:
                        for gid in alive:
                            g = self.ghosts[gid]
                            if len(g.cbba_agent.bundle) == 0:
                                d_rl = dijkstra_multi(g.world, (g.y, g.x), all_targets) if all_targets else {}
                                g.cbba_agent._phase1(g, deduped_pool, d_rl)

        # 2. Advance Pac-Man
        prev_pow = self.player.powered
        v_grid_s = 0.0
        if self.speed_mode == "calibrated":
            pac_wx, pac_wy, pac_vx, pac_vy = self.player.step_continuous(dt=self.dt_step, ghosts=self.ghosts, powered=self.player.powered)
            v_grid_s = math.hypot(pac_vx, pac_vy) / self.scale
            # Pellet pickup & power consumption matching game_engine.py
            eat_radius = 0.60 * self.scale
            px_m = (self.player.x - self.world.width / 2.0) * self.scale
            py_m = (self.player.y - self.world.height / 2.0) * self.scale

            p_path = getattr(self.player, "path_this_frame", None)

            def _dist(tx, ty):
                if p_path and len(p_path) == 2:
                    (x1, y1), (x2, y2) = p_path
                    dx, dy = x2 - x1, y2 - y1
                    l2 = dx * dx + dy * dy
                    if l2 > 1e-8:
                        t = max(0.0, min(1.0, ((tx - x1) * dx + (ty - y1) * dy) / l2))
                        return math.hypot(tx - (x1 + t * dx), ty - (y1 + t * dy))
                return math.hypot(px_m - tx, py_m - ty)

            eaten_p = []
            for pt in list(self.world.pellets):
                wx = (pt[0] - self.world.width / 2.0) * self.scale
                wy = (pt[1] - self.world.height / 2.0) * self.scale
                if _dist(wx, wy) <= eat_radius:
                    eaten_p.append(pt)
            for pt in eaten_p:
                if pt in self.world.pellets:
                    self.world.pellets.remove(pt)
                    self.player.score += 10
            if eaten_p:
                self.world._update_pellet_arrays()

            eaten_pp = []
            for pt in list(self.world.power_pellets):
                wx = (pt[0] - self.world.width / 2.0) * self.scale
                wy = (pt[1] - self.world.height / 2.0) * self.scale
                if _dist(wx, wy) <= eat_radius:
                    eaten_pp.append(pt)
            for pt in eaten_pp:
                if pt in self.world.power_pellets:
                    self.world.power_pellets.remove(pt)
                    self.player.score += 50
                    self.player.powered = True
                    self.player.power_timer = self.power_duration_steps
            if eaten_pp:
                self.world._update_pellet_arrays()
        else:
            self.player.update(self.ghosts)

        if not prev_pow and self.player.powered:
            self.power_pellets_eaten_total += 1
            self.events.append({"time": self.sim_time, "event": "POWER_EATEN", "pos": (self.player.x, self.player.y)})

        # 3. Advance Ghost agents
        powered_flag = self.player.powered
        for gid, ghost in self.ghosts.items():
            if ghost.dead:
                continue
            ghost.power_pellets_converted_this_frame = 0
            if self.speed_mode == "calibrated":
                is_hunting = (getattr(ghost, "known_pacman", None) is not None or getattr(ghost, "_is_striking", False)) and not ghost.pacman_powered
                base_v = 0.95 if is_hunting else 0.80
                if ghost.pacman_powered:
                    base_v = 0.52
                ghost.max_speed = (base_v / self.scale) * self.dt_step
                ghost.update((self.player.y, self.player.x), powered_flag, self.ghosts, speed_mult=1.0)

                # Defusal check matching game_engine.py: 0.85 * scale
                gx_m = (ghost.x - self.world.width / 2.0) * self.scale
                gy_m = (ghost.y - self.world.height / 2.0) * self.scale
                defuse_rad = 0.85 * self.scale
                for pt in list(self.world.power_pellets):
                    wx = (pt[0] - self.world.width / 2.0) * self.scale
                    wy = (pt[1] - self.world.height / 2.0) * self.scale
                    if math.hypot(wx - gx_m, wy - gy_m) <= defuse_rad:
                        self.world.power_pellets.remove(pt)
                        if pt not in self.world.pellets:
                            self.world.pellets.append(pt)
                        self.world._update_pellet_arrays()
                        ghost.power_pellets_converted_this_frame += 1
                        ghost.known_power_pellets.discard(pt)
                        ghost.known_pellets.add(pt)
                        ghost._broadcast([("power_eaten", pt), ("pellet", pt)], self.ghosts)
                        break
            else:
                spd = 0.65 if ghost.pacman_powered else getattr(ghost, "current_speed_mult", 1.0)
                ghost.update((self.player.y, self.player.x), powered_flag, self.ghosts, speed_mult=spd)

            conv = getattr(ghost, "power_pellets_converted_this_frame", 0)
            if conv > 0:
                self.power_pellets_defused_total += conv
                self.events.append({"time": self.sim_time, "event": "POWER_DEFUSED", "gid": gid, "pos": (ghost.x, ghost.y)})

        # 4. Agent Collisions (swept-path interpolation matching pacman.py)
        if not self.player.dead:
            for gid, ghost in list(self.ghosts.items()):
                if ghost.dead:
                    continue
                if abs(ghost.x - self.player.x) > 2.0 or abs(ghost.y - self.player.y) > 2.0:
                    continue

                col_rad = self.player.radius + ghost.radius + 0.15
                collided = False
                p_path = getattr(self.player, "path_this_frame", [(self.player.x, self.player.y)])
                g_path = getattr(ghost, "path_this_frame", [(ghost.x, ghost.y)])
                samples = max(1, len(p_path) - 1, len(g_path) - 1) * 3 + 1

                for step_i in range(samples):
                    t = step_i / max(1, samples - 1)
                    px = p_path[0][0] * (1 - t) + p_path[-1][0] * t
                    py = p_path[0][1] * (1 - t) + p_path[-1][1] * t
                    gx = g_path[0][0] * (1 - t) + g_path[-1][0] * t
                    gy = g_path[0][1] * (1 - t) + g_path[-1][1] * t
                    if math.hypot(gx - px, gy - py) < col_rad:
                        collided = True
                        break

                if collided:
                    if self.player.powered:
                        ghost.kill()
                        self.ghosts_eaten_total += 1
                        self.player.score += 200
                        self.events.append({"time": self.sim_time, "event": "GHOST_EATEN", "gid": gid, "pos": (ghost.x, ghost.y)})
                    else:
                        self.player.die()
                        self.game_over = True
                        self.outcome = "GHOSTS_WIN"
                        self.events.append({"time": self.sim_time, "event": "PACMAN_KILLED", "gid": gid, "pos": (self.player.x, self.player.y)})
                        break

        # 5. Check Win Conditions
        if len(self.world.pellets) == 0 and len(self.world.power_pellets) == 0:
            self.game_over = True
            self.outcome = "PACMAN_WIN"
        elif all(g.dead for g in self.ghosts.values()):
            self.game_over = True
            self.outcome = "PACMAN_WIN"

        # Record telemetry
        self.pacman_traj.append((self.sim_time, float(self.player.x), float(self.player.y)))
        spd_scale = self.dt_step if self.speed_mode == "calibrated" else 1.0
        if self.speed_mode == "calibrated":
            self.pacman_speeds.append(v_grid_s)
        else:
            self.pacman_speeds.append(math.hypot(self.player.vx, self.player.vy))

        live_ghost_speeds = [math.hypot(g.vx, g.vy) / spd_scale for g in self.ghosts.values() if not g.dead]
        if live_ghost_speeds:
            self.ghost_speeds.append(float(np.mean(live_ghost_speeds)))

        for gid, g in self.ghosts.items():
            if not g.dead:
                self.ghost_trajs[gid].append((self.sim_time, float(g.x), float(g.y)))

        self.pellet_history.append((self.sim_time, len(self.world.pellets) + len(self.world.power_pellets)))
        self.score_history.append((self.sim_time, self.player.score))

    def run(self, max_time: float = 120.0):
        max_steps = int(max_time / self.dt_step)
        for _ in range(max_steps):
            if self.game_over:
                break
            self.step()

        if not self.game_over:
            self.game_over = True
            self.outcome = "TIMEOUT"

        return {
            "name": "pacman.py (2D Kinematic)",
            "outcome": self.outcome,
            "duration_s": self.sim_time,
            "steps": self.frame_counter,
            "final_score": self.player.score,
            "pellets_left": len(self.world.pellets),
            "power_left": len(self.world.power_pellets),
            "pellets_eaten": self.initial_pellets_count - len(self.world.pellets),
            "power_eaten": self.power_pellets_eaten_total,
            "power_defused": self.power_pellets_defused_total,
            "ghosts_eaten": self.ghosts_eaten_total,
            "ghosts_alive": sum(1 for g in self.ghosts.values() if not g.dead),
            "avg_pacman_speed": float(np.mean(self.pacman_speeds)) if self.pacman_speeds else 0.0,
            "avg_ghost_speed": float(np.mean(self.ghost_speeds)) if self.ghost_speeds else 0.0,
            "pacman_traj": self.pacman_traj,
            "ghost_trajs": self.ghost_trajs,
            "pellet_history": self.pellet_history,
            "score_history": self.score_history,
            "events": self.events,
        }


# =============================================================================
# Headless Runner 2: MuJoCo Continuous Multi-Agent Physics Simulation
# =============================================================================
class HeadlessMujocoRunner:
    """Runs Native MuJoCo continuous simulation with SvanM2 quadrupeds headlessly."""

    def __init__(
        self,
        stage_idx: int,
        seed: int,
        scale: float,
        policy_ckpt: str,
        ghost_ckpt: int,
        enable_rl: bool,
        world,
        grid: np.ndarray,
        player_start: Tuple[float, float],
        ghost_starts: List[Tuple[float, float]],
        dt_step: float = 0.020,  # 50 Hz policy step
    ):
        self.scale = scale
        self.dt_step = dt_step
        self.engine = PacmanGameEngine(
            stage=stage_idx,
            seed=seed,
            scale=scale,
            policy_ckpt=policy_ckpt,
            ghost_ckpt=ghost_ckpt,
            enable_rl=enable_rl,
            world=world,
            grid=grid,
            player_start_grid=player_start,
            ghost_starts_grid=ghost_starts,
        )

        self.initial_pellets_count = len(self.engine.active_pellets)
        self.initial_power_count = len(self.engine.active_power)

        self.sim_time = 0.0
        self.steps = 0

        # Telemetry traces
        self.pacman_traj: List[Tuple[float, float, float]] = []  # (t, gx, gy)
        self.pacman_traj_metric: List[Tuple[float, float, float]] = []  # (t, wx, wy)
        self.ghost_trajs: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(self.engine.arena.n_ghosts)}
        self.pacman_speeds: List[float] = []
        self.ghost_speeds: List[float] = []
        self.pellet_history: List[Tuple[float, int]] = []
        self.score_history: List[Tuple[float, int]] = []
        self.events: List[Dict[str, Any]] = []

        self.power_pellets_defused_total = 0
        self.power_pellets_eaten_total = 0
        self.ghosts_eaten_total = 0
        self._prev_pow_count = self.initial_power_count
        self._prev_powered = False

    def step(self):
        if self.engine.game_over:
            return

        prev_active_power = len(self.engine.active_power)
        prev_is_pow = self.engine.is_pacman_powered

        self.engine.step_simulation()
        self.steps += 1
        self.sim_time += self.dt_step

        # Detect power events
        if not prev_is_pow and self.engine.is_pacman_powered:
            self.power_pellets_eaten_total += 1
            pac_wx, pac_wy, _ = self.engine.pacman_position
            gx, gy = self.engine.arena.to_grid(pac_wx, pac_wy)
            self.events.append({"time": self.sim_time, "event": "POWER_EATEN", "pos": (float(gx), float(gy))})

        # Convert continuous positions to grid coordinates (1:1 with pacman.py)
        pac_wx, pac_wy, _ = self.engine.pacman_position
        pac_gx, pac_gy = self.engine.arena.to_grid(pac_wx, pac_wy)

        self.pacman_traj.append((self.sim_time, float(pac_gx), float(pac_gy)))
        self.pacman_traj_metric.append((self.sim_time, float(pac_wx), float(pac_wy)))

        # Velocity tracking
        pac_dof = self.engine.pac_dof_adr
        vx = self.engine.data.qvel[pac_dof]
        vy = self.engine.data.qvel[pac_dof + 1]
        v_metric = math.hypot(vx, vy)
        v_grid = v_metric / self.scale
        self.pacman_speeds.append(v_grid)

        # Track ghost positions and physical speeds
        current_ghost_speeds = []
        for gid, runner in enumerate(self.engine.runners):
            if not self.engine.hive.ghosts[gid].dead:
                g_wx, g_wy, _ = runner.position
                gx, gy = self.engine.arena.to_grid(g_wx, g_wy)
                self.ghost_trajs[gid].append((self.sim_time, float(gx), float(gy)))
                vx_g, vy_g = runner.cmd_base[0], runner.cmd_base[1]
                current_ghost_speeds.append(math.hypot(vx_g, vy_g) / self.scale)

        if current_ghost_speeds:
            self.ghost_speeds.append(float(np.mean(current_ghost_speeds)))

        self.pellet_history.append((self.sim_time, len(self.engine.active_pellets) + len(self.engine.active_power)))
        self.score_history.append((self.sim_time, self.engine.score))

    def run(self, max_time: float = 120.0):
        max_steps = int(max_time / self.dt_step)
        for _ in range(max_steps):
            if self.engine.game_over:
                break
            self.step()

        if not self.engine.game_over:
            self.engine.game_over = True
            self.engine.outcome = "TIMEOUT"

        # Tally power conversions and ghost eliminations
        for g in self.engine.hive.ghosts.values():
            self.power_pellets_defused_total += getattr(g, "power_pellets_converted_this_frame", 0)

        dead_ghosts = sum(1 for g in self.engine.hive.ghosts.values() if g.dead)

        return {
            "name": "MuJoCo (Native Multi-Agent Physics)",
            "outcome": self.engine.outcome,
            "duration_s": self.sim_time,
            "steps": self.steps,
            "final_score": self.engine.score,
            "pellets_left": len(self.engine.active_pellets),
            "power_left": len(self.engine.active_power),
            "pellets_eaten": max(0, self.initial_pellets_count - len(self.engine.active_pellets)),
            "power_eaten": self.power_pellets_eaten_total,
            "power_defused": self.power_pellets_defused_total,
            "ghosts_eaten": dead_ghosts,
            "ghosts_alive": self.engine.hive.alive_ghost_count,
            "avg_pacman_speed": float(np.mean(self.pacman_speeds)) if self.pacman_speeds else 0.0,
            "avg_ghost_speed": float(np.mean(self.ghost_speeds)) if self.ghost_speeds else 0.0,
            "pacman_traj": self.pacman_traj,
            "pacman_traj_metric": self.pacman_traj_metric,
            "ghost_trajs": self.ghost_trajs,
            "pellet_history": self.pellet_history,
            "score_history": self.score_history,
            "events": self.events,
        }


# =============================================================================
# Comparative Analytics & Divergence Metrics
# =============================================================================
def compare_runs(mj_res: Dict[str, Any], py_res: Dict[str, Any], scale: float = 1.5) -> Dict[str, Any]:
    """Computes exact comparative divergence metrics between MuJoCo and pacman.py."""
    outcomes_agree = (mj_res["outcome"] == py_res["outcome"])

    # 1. Trajectory Divergence
    traj_mj = mj_res["pacman_traj"]
    traj_py = py_res["pacman_traj"]

    min_len = min(len(traj_mj), len(traj_py))
    pointwise_distances_grid = []
    first_divergence_time = None
    divergence_threshold = 1.5  # grid units

    for i in range(min_len):
        t_mj, x_mj, y_mj = traj_mj[i]
        t_py, x_py, y_py = traj_py[i]
        dist = math.hypot(x_mj - x_py, y_mj - y_py)
        pointwise_distances_grid.append(dist)

        if first_divergence_time is None and dist >= divergence_threshold:
            first_divergence_time = t_mj

    mean_traj_divergence_grid = float(np.mean(pointwise_distances_grid)) if pointwise_distances_grid else 0.0
    max_traj_divergence_grid = float(np.max(pointwise_distances_grid)) if pointwise_distances_grid else 0.0
    mean_traj_divergence_m = mean_traj_divergence_grid * scale
    max_traj_divergence_m = max_traj_divergence_grid * scale

    # Discrete Fréchet Distance (geometric curve comparison)
    pts_mj = np.array([[x, y] for _, x, y in traj_mj])
    pts_py = np.array([[x, y] for _, x, y in traj_py])
    frechet_dist_grid = discrete_frechet_distance(pts_mj, pts_py)
    frechet_dist_m = frechet_dist_grid * scale

    # Ghost Swarm Trajectory Divergence
    ghost_mean_dists_grid = {}
    for gid in mj_res["ghost_trajs"]:
        if gid in py_res["ghost_trajs"]:
            g_mj = mj_res["ghost_trajs"][gid]
            g_py = py_res["ghost_trajs"][gid]
            g_len = min(len(g_mj), len(g_py))
            if g_len > 0:
                dists = [math.hypot(g_mj[k][1] - g_py[k][1], g_mj[k][2] - g_py[k][2]) for k in range(g_len)]
                ghost_mean_dists_grid[gid] = float(np.mean(dists))

    avg_ghost_divergence_grid = float(np.mean(list(ghost_mean_dists_grid.values()))) if ghost_mean_dists_grid else 0.0

    # 2. Survival & Scoring Deltas
    dur_mj = mj_res["duration_s"]
    dur_py = py_res["duration_s"]
    dur_diff = dur_mj - dur_py

    score_diff = mj_res["final_score"] - py_res["final_score"]
    pellets_diff = mj_res["pellets_eaten"] - py_res["pellets_eaten"]

    return {
        "outcomes_agree": outcomes_agree,
        "outcome_mujoco": mj_res["outcome"],
        "outcome_pacman_py": py_res["outcome"],
        "duration_mujoco_s": dur_mj,
        "duration_pacman_py_s": dur_py,
        "duration_delta_s": dur_diff,
        "score_mujoco": mj_res["final_score"],
        "score_pacman_py": py_res["final_score"],
        "score_delta": score_diff,
        "pellets_eaten_mujoco": mj_res["pellets_eaten"],
        "pellets_eaten_pacman_py": py_res["pellets_eaten"],
        "pellets_delta": pellets_diff,
        "power_eaten_mujoco": mj_res["power_eaten"],
        "power_eaten_pacman_py": py_res["power_eaten"],
        "power_defused_mujoco": mj_res["power_defused"],
        "power_defused_pacman_py": py_res["power_defused"],
        "ghosts_eaten_mujoco": mj_res["ghosts_eaten"],
        "ghosts_eaten_pacman_py": py_res["ghosts_eaten"],
        "mean_pacman_divergence_grid": mean_traj_divergence_grid,
        "mean_pacman_divergence_m": mean_traj_divergence_m,
        "max_pacman_divergence_grid": max_traj_divergence_grid,
        "max_pacman_divergence_m": max_traj_divergence_m,
        "frechet_distance_grid": frechet_dist_grid,
        "frechet_distance_m": frechet_dist_m,
        "first_divergence_time_s": first_divergence_time,
        "mean_ghost_divergence_grid": avg_ghost_divergence_grid,
        "mean_ghost_divergence_m": avg_ghost_divergence_grid * scale,
        "pacman_avg_speed_mujoco": mj_res["avg_pacman_speed"],
        "pacman_avg_speed_pacman_py": py_res["avg_pacman_speed"],
        "ghost_avg_speed_mujoco": mj_res["avg_ghost_speed"],
        "ghost_avg_speed_pacman_py": py_res["avg_ghost_speed"],
        "pointwise_distances_grid": pointwise_distances_grid,
    }


# =============================================================================
# Diagnostic Visualization Plotting (Matplotlib)
# =============================================================================
def plot_comparative_trajectories(
    map_data: Dict[str, Any],
    mj_res: Dict[str, Any],
    py_res: Dict[str, Any],
    comparison: Dict[str, Any],
    plot_path: str,
    scale: float = 1.5,
):
    """Generates an aesthetic 4-panel comparative telemetry plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 12), dpi=140)
    fig.patch.set_facecolor("#0b0e14")

    # Panel 1: Top-Down Spatial Arena & Trajectory Overlay
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#0f131a")

    world = map_data["world_mj"]
    grid = map_data["grid_mj"]
    rows, cols = grid.shape

    # Draw Obstacle boundary lines / capsules
    for obs in getattr(world, "obstacles", []):
        caps = [obs] if hasattr(obs, "x1") else getattr(obs, "capsules", [])
        for c in caps:
            ax1.plot([c.x1, c.x2], [c.y1, c.y2], color="#3d4966", lw=3.5, solid_capstyle="round", alpha=0.85)

    # Pellets
    if getattr(world, "pellets", None):
        px = [p[0] for p in world.pellets]
        py = [p[1] for p in world.pellets]
        ax1.scatter(px, py, color="#8a95aa", s=8, alpha=0.5, label="Pellets")

    # Power Pellets
    if getattr(world, "power_pellets", None):
        ppx = [p[0] for p in world.power_pellets]
        ppy = [p[1] for p in world.power_pellets]
        ax1.scatter(ppx, ppy, color="#00ffcc", s=45, marker="D", edgecolors="white", label="Power Pellets")

    # Pac-Man trajectories
    mj_pts = np.array([[x, y] for _, x, y in mj_res["pacman_traj"]])
    py_pts = np.array([[x, y] for _, x, y in py_res["pacman_traj"]])

    if len(mj_pts) > 0:
        ax1.plot(mj_pts[:, 0], mj_pts[:, 1], color="#ffb703", lw=2.2, label=f"Pacman MuJoCo ({mj_res['outcome']})")
        ax1.scatter(mj_pts[-1, 0], mj_pts[-1, 1], color="#ffb703", s=70, marker="*", edgecolors="black", zorder=5)

    if len(py_pts) > 0:
        ax1.plot(py_pts[:, 0], py_pts[:, 1], color="#00b4d8", lw=1.8, linestyle="--", label=f"Pacman pacman.py ({py_res['outcome']})")
        ax1.scatter(py_pts[-1, 0], py_pts[-1, 1], color="#00b4d8", s=60, marker="X", edgecolors="black", zorder=5)

    # Spawn points
    ps = map_data["player_start"]
    ax1.scatter(ps[1], ps[0], color="#ffe600", s=110, marker="o", edgecolors="black", linewidth=1.5, label="Pac-Man Spawn", zorder=6)

    for i, gs in enumerate(map_data["ghost_starts"]):
        lbl = "Ghost Spawns" if i == 0 else None
        ax1.scatter(gs[1], gs[0], color="#ff0054", s=80, marker="s", edgecolors="black", label=lbl, zorder=6)

    ax1.set_title("1. Spatial Trajectories Overlay (Grid Coordinates)", color="#ffffff", fontsize=12, pad=10, fontweight="bold")
    ax1.set_xlim(-1, cols + 1)
    ax1.set_ylim(rows + 1, -1)  # Inverted y for top-down grid matching screen
    ax1.tick_params(colors="#8a95aa")
    ax1.legend(loc="upper right", facecolor="#141923", edgecolor="#2d3748", labelcolor="#e2e8f0", fontsize=8)

    # Panel 2: Pac-Man Trajectory Divergence d(t) over Time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#0f131a")

    times = [t for t, _, _ in mj_res["pacman_traj"][: len(comparison["pointwise_distances_grid"])]]
    dists_grid = comparison["pointwise_distances_grid"]
    dists_m = [d * scale for d in dists_grid]

    ax2.plot(times, dists_grid, color="#e63946", lw=2.0, label="Divergence ||p_mj(t) - p_py(t)|| (cells)")
    ax2.axhline(1.5, color="#f4a261", linestyle=":", lw=1.5, label="Corridor Branch Threshold (1.5 cells)")
    if comparison["first_divergence_time_s"] is not None:
        ax2.axvline(comparison["first_divergence_time_s"], color="#00f5d4", linestyle="--", lw=1.2, label=f"First Branch @ {comparison['first_divergence_time_s']:.1f}s")

    ax2.set_title("2. Trajectory Divergence Over Time", color="#ffffff", fontsize=12, pad=10, fontweight="bold")
    ax2.set_xlabel("Simulated Time (seconds)", color="#8a95aa")
    ax2.set_ylabel("Separation Distance (grid cells)", color="#8a95aa")
    ax2.tick_params(colors="#8a95aa")
    ax2.grid(True, color="#1e2638", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", facecolor="#141923", edgecolor="#2d3748", labelcolor="#e2e8f0", fontsize=8)

    # Panel 3: Score & Pellet Clearance Progression
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#0f131a")

    t_sc_mj = [t for t, _ in mj_res["score_history"]]
    sc_mj = [s for _, s in mj_res["score_history"]]
    t_sc_py = [t for t, _ in py_res["score_history"]]
    sc_py = [s for _, s in py_res["score_history"]]

    ax3.plot(t_sc_mj, sc_mj, color="#ffb703", lw=2.2, label="Score: MuJoCo")
    ax3.plot(t_sc_py, sc_py, color="#00b4d8", lw=2.0, linestyle="--", label="Score: pacman.py")

    ax3.set_title("3. Score Accumulation Timeline", color="#ffffff", fontsize=12, pad=10, fontweight="bold")
    ax3.set_xlabel("Simulated Time (seconds)", color="#8a95aa")
    ax3.set_ylabel("Game Score (points)", color="#8a95aa")
    ax3.tick_params(colors="#8a95aa")
    ax3.grid(True, color="#1e2638", linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", facecolor="#141923", edgecolor="#2d3748", labelcolor="#e2e8f0", fontsize=8)

    # Panel 4: Pellets Remaining Countdown
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#0f131a")

    t_pel_mj = [t for t, _ in mj_res["pellet_history"]]
    pel_mj = [p for _, p in mj_res["pellet_history"]]
    t_pel_py = [t for t, _ in py_res["pellet_history"]]
    pel_py = [p for _, p in py_res["pellet_history"]]

    ax4.plot(t_pel_mj, pel_mj, color="#ffb703", lw=2.2, label="Pellets Left: MuJoCo")
    ax4.plot(t_pel_py, pel_py, color="#00b4d8", lw=2.0, linestyle="--", label="Pellets Left: pacman.py")

    ax4.set_title("4. Pellets Remaining Countdown", color="#ffffff", fontsize=12, pad=10, fontweight="bold")
    ax4.set_xlabel("Simulated Time (seconds)", color="#8a95aa")
    ax4.set_ylabel("Active Pellets Remaining", color="#8a95aa")
    ax4.tick_params(colors="#8a95aa")
    ax4.grid(True, color="#1e2638", linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right", facecolor="#141923", edgecolor="#2d3748", labelcolor="#e2e8f0", fontsize=8)

    plt.suptitle(
        f"Sim-to-Sim Reality Gap: MuJoCo (SvanM2 Quadrupeds) vs pacman.py (2D Kinematics)\n"
        f"Map: Stage {map_data['stage_cfg'].rows}x{map_data['stage_cfg'].cols} | Fréchet Divergence: {comparison['frechet_distance_m']:.2f}m ({comparison['frechet_distance_grid']:.2f} cells) | "
        f"Outcomes Agree: {comparison['outcomes_agree']}",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [Plot] Diagnostic comparative telemetry plot saved: {plot_path}")


# =============================================================================
# Formatted Terminal Report Display
# =============================================================================
def print_comparison_table(res_mj: Dict[str, Any], res_py: Dict[str, Any], comp: Dict[str, Any], seed: int, stage: int):
    """Prints a beautiful side-by-side terminal comparison table."""
    border = "=" * 80
    sub_border = "-" * 80

    print("\n" + border)
    print(f"       SIMULATION COMPARISON REPORT: MUJOCO vs PACMAN.PY (Stage {stage}, Seed {seed})")
    print(border)
    print(f"{'METRIC':<36} | {'MUJOCO (PHYSICS)':<18} | {'PACMAN.PY (2D)':<18}")
    print(sub_border)

    match_str = "YES" if comp["outcomes_agree"] else "NO (DIVERGED)"
    print(f"{'Game Outcome':<36} | {comp['outcome_mujoco']:<18} | {comp['outcome_pacman_py']:<18}")
    print(f"{'Outcomes Agreement':<36} | {match_str:<18} | {match_str:<18}")
    print(f"{'Simulated Duration':<36} | {comp['duration_mujoco_s']:>15.2f} s | {comp['duration_pacman_py_s']:>15.2f} s")
    print(f"{'Simulation Duration Delta':<36} | {comp['duration_delta_s']:>+15.2f} s | {'-':>18}")
    print(sub_border)
    print(f"{'Final Score':<36} | {comp['score_mujoco']:>15d}   | {comp['score_pacman_py']:>15d}  ")
    print(f"{'Regular Pellets Eaten':<36} | {comp['pellets_eaten_mujoco']:>15d}   | {comp['pellets_eaten_pacman_py']:>15d}  ")
    print(f"{'Power Pellets Consumed':<36} | {comp['power_eaten_mujoco']:>15d}   | {comp['power_eaten_pacman_py']:>15d}  ")
    print(f"{'Power Pellets Defused (CONVERT)':<36} | {comp['power_defused_mujoco']:>15d}   | {comp['power_defused_pacman_py']:>15d}  ")
    print(f"{'Ghosts Eliminated':<36} | {comp['ghosts_eaten_mujoco']:>15d}   | {comp['ghosts_eaten_pacman_py']:>15d}  ")
    print(sub_border)
    print(f"{'Mean Trajectory Divergence':<36} | {comp['mean_pacman_divergence_m']:>13.2f} m | {comp['mean_pacman_divergence_grid']:>11.2f} cells")
    print(f"{'Max Trajectory Divergence':<36} | {comp['max_pacman_divergence_m']:>13.2f} m | {comp['max_pacman_divergence_grid']:>11.2f} cells")
    print(f"{'Fréchet Curve Divergence':<36} | {comp['frechet_distance_m']:>13.2f} m | {comp['frechet_distance_grid']:>11.2f} cells")
    first_branch_str = f"{comp['first_divergence_time_s']:.2f} s" if comp["first_divergence_time_s"] is not None else "None"
    print(f"{'Time of First Corridor Branch':<36} | {first_branch_str:>18} | {'-':>18}")
    print(f"{'Mean Ghost Swarm Divergence':<36} | {comp['mean_ghost_divergence_m']:>13.2f} m | {comp['mean_ghost_divergence_grid']:>11.2f} cells")
    print(sub_border)
    print(f"{'Pac-Man Avg Realized Speed':<36} | {comp['pacman_avg_speed_mujoco']:>13.2f} c/s | {comp['pacman_avg_speed_pacman_py']:>11.2f} c/s")
    print(f"{'Ghost Swarm Avg Realized Speed':<36} | {comp['ghost_avg_speed_mujoco']:>13.2f} c/s | {comp['ghost_avg_speed_pacman_py']:>11.2f} c/s")
    print(border + "\n")


# =============================================================================
# Main Entrypoint
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless Comparison: Native MuJoCo Multi-Agent Physics vs pacman.py 2D Kinematics."
    )
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Curriculum stage (1 to 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for procedural maze generation.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of comparative runs across consecutive seeds.")
    parser.add_argument("--scale", type=float, default=1.5, help="Metric scale factor (meters per grid cell, default: 1.5).")
    parser.add_argument("--max_time", type=float, default=60.0, help="Maximum duration per run in simulated seconds.")
    parser.add_argument("--policy", type=str, default="checkpoints/svanm2_flat/svanm2_flat.pt", help="Path to SvanM2 flat locomotion policy.")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Ghost MAPPO checkpoint (-1 for latest).")
    parser.add_argument("--no_rl", action="store_true", help="Run in purely heuristic CBBA mode without neural network.")
    parser.add_argument(
        "--mode",
        type=str,
        default="calibrated",
        choices=["calibrated", "raw"],
        help="pacman.py pacing mode: 'calibrated' (matches continuous metric speed 1.40 m/s for direct physics gap benchmark) or 'raw' (original discrete 1.0 cell/step kinematics).",
    )
    parser.add_argument("--plot", type=str, default=None, help="Save diagnostic 4-panel trajectory plot to this image path.")
    parser.add_argument("--output", type=str, default=None, help="Save full JSON comparison report to this file path.")
    return parser.parse_args()


def main():
    args = parse_args()
    policy_path = _REPO_ROOT / args.policy
    assert policy_path.exists(), f"Policy checkpoint not found at {policy_path}"

    print("=================================================================")
    print("      HEADLESS BENCHMARK: MUJOCO vs PACMAN.PY COMPARISON         ")
    print("=================================================================")
    print(f"  Stage:                  {args.stage}")
    print(f"  Start Seed:             {args.seed}")
    print(f"  Total Comparative Runs: {args.num_runs}")
    print(f"  Scale Factor:           {args.scale:.2f} m/unit")
    print(f"  Max Run Time:           {args.max_time:.1f} s")
    print(f"  Locomotion Policy:      {policy_path.name}")
    print(f"  Ghost Autonomy Mode:    {'Heuristic CBBA' if args.no_rl else 'MAPPO RL + CBBA'}")
    print(f"  Plot Output:            {args.plot or 'None'}")
    print(f"  JSON Output:            {args.output or 'None'}")
    print("=================================================================")

    all_comparisons = []
    run_summaries = []

    for run_i in range(args.num_runs):
        cur_seed = args.seed + run_i
        print(f"\n>>> Running Comparative Episode [{run_i + 1}/{args.num_runs}] (Seed {cur_seed})...")

        # 1. Setup exact shared map and spawn points
        map_data = setup_shared_map(stage_idx=args.stage, seed=cur_seed)
        n_pel = map_data["total_pellets"]
        n_pow = map_data["total_power"]
        p_start = map_data["player_start"]
        g_starts = map_data["ghost_starts"]

        print(f"  • Map generated: {map_data['stage_cfg'].rows}x{map_data['stage_cfg'].cols} cells")
        print(f"  • Shared Pellets: {n_pel} regular, {n_pow} power")
        print(f"  • Pac-Man start: (row={p_start[0]}, col={p_start[1]})")
        print(f"  • Ghost starts: {len(g_starts)} agents spawned identically")

        # 2. Run MuJoCo simulation headlessly
        t0_mj = time.time()
        print("  [1/2] Executing Native MuJoCo 3D Quadruped Simulation (50 Hz)...")
        mj_runner = HeadlessMujocoRunner(
            stage_idx=args.stage,
            seed=cur_seed,
            scale=args.scale,
            policy_ckpt=str(policy_path),
            ghost_ckpt=args.checkpoint,
            enable_rl=not args.no_rl,
            world=map_data["world_mj"],
            grid=map_data["grid_mj"],
            player_start=p_start,
            ghost_starts=g_starts,
        )
        res_mj = mj_runner.run(max_time=args.max_time)
        t_wall_mj = time.time() - t0_mj
        print(f"        -> Finished in {t_wall_mj:.2f}s wall time! Outcome: {res_mj['outcome']} ({res_mj['duration_s']:.1f}s sim)")

        # 3. Run pacman.py headlessly
        t0_py = time.time()
        print(f"  [2/2] Executing pacman.py 2D Kinematic Simulation (50 Hz, mode: {args.mode})...")
        py_runner = HeadlessPacmanPyRunner(
            grid=map_data["grid_py"],
            player_start=p_start,
            ghost_starts=g_starts,
            world=map_data["world_py"],
            enable_rl=not args.no_rl,
            ghost_ckpt=args.checkpoint,
            scale=args.scale,
            speed_mode=args.mode,
        )
        res_py = py_runner.run(max_time=args.max_time)
        t_wall_py = time.time() - t0_py
        print(f"        -> Finished in {t_wall_py:.2f}s wall time! Outcome: {res_py['outcome']} ({res_py['duration_s']:.1f}s sim)")

        # 4. Compute comparative divergence
        comp = compare_runs(res_mj, res_py, scale=args.scale)
        all_comparisons.append(comp)

        run_summaries.append({
            "seed": cur_seed,
            "stage": args.stage,
            "mujoco": {
                "outcome": res_mj["outcome"],
                "duration_s": res_mj["duration_s"],
                "score": res_mj["final_score"],
                "pellets_eaten": res_mj["pellets_eaten"],
                "power_eaten": res_mj["power_eaten"],
                "power_defused": res_mj["power_defused"],
                "ghosts_eaten": res_mj["ghosts_eaten"],
            },
            "pacman_py": {
                "outcome": res_py["outcome"],
                "duration_s": res_py["duration_s"],
                "score": res_py["final_score"],
                "pellets_eaten": res_py["pellets_eaten"],
                "power_eaten": res_py["power_eaten"],
                "power_defused": res_py["power_defused"],
                "ghosts_eaten": res_py["ghosts_eaten"],
            },
            "divergence": {
                "outcomes_agree": comp["outcomes_agree"],
                "duration_delta_s": comp["duration_delta_s"],
                "score_delta": comp["score_delta"],
                "mean_pacman_divergence_m": comp["mean_pacman_divergence_m"],
                "frechet_distance_m": comp["frechet_distance_m"],
                "first_divergence_time_s": comp["first_divergence_time_s"],
            },
        })

        print_comparison_table(res_mj, res_py, comp, seed=cur_seed, stage=args.stage)

        # Plot first run if requested
        if args.plot and run_i == 0:
            plot_comparative_trajectories(
                map_data=map_data,
                mj_res=res_mj,
                py_res=res_py,
                comparison=comp,
                plot_path=args.plot,
                scale=args.scale,
            )

    # Multi-Run Aggregate Summary
    if args.num_runs > 1:
        agree_count = sum(1 for c in all_comparisons if c["outcomes_agree"])
        agree_pct = 100.0 * agree_count / args.num_runs
        mean_dur_delta = np.mean([abs(c["duration_delta_s"]) for c in all_comparisons])
        mean_frechet = np.mean([c["frechet_distance_m"] for c in all_comparisons])
        mean_diverge = np.mean([c["mean_pacman_divergence_m"] for c in all_comparisons])

        print("=================================================================")
        print("               MULTI-RUN STATISTICAL BENCHMARK SUMMARY           ")
        print("=================================================================")
        print(f"  Runs Tested:                 {args.num_runs}")
        print(f"  Outcome Agreement Rate:      {agree_pct:.1f}% ({agree_count}/{args.num_runs} runs)")
        print(f"  Mean Duration Difference:    {mean_dur_delta:.2f} s")
        print(f"  Mean Fréchet Divergence:     {mean_frechet:.2f} m")
        print(f"  Mean Pointwise Separation:   {mean_diverge:.2f} m")
        print("=================================================================")

    # Export JSON report
    if args.output:
        report_data = {
            "config": {
                "stage": args.stage,
                "start_seed": args.seed,
                "num_runs": args.num_runs,
                "scale": args.scale,
                "max_time": args.max_time,
                "enable_rl": not args.no_rl,
            },
            "runs": run_summaries,
        }
        with open(args.output, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"  [JSON] Complete comparison report saved to: {args.output}")


if __name__ == "__main__":
    main()
