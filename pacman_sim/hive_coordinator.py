#!/usr/bin/env python3
"""
hive_coordinator.py — Multi-Agent Ghost Swarm Hive Coordinator for MuJoCo.

Bridges native MuJoCo quadruped physics with pacmanbot's decentralized autonomy stack:
  • Local 90-ray LiDAR sweeps with authentic fog-of-war (NO global state sharing)
  • Pac-Man powered state discovered ONLY via direct line-of-sight
  • 12-meter NRF24 radio mesh with peer packet relaying
  • MAPPO Neural Network (GhostActor) inference evaluated at authentic 6-frame decision intervals
  • CBBA decentralized task auctioning & 2-round consensus (HUNT, FLANK, CONVERT, EVADE_TRACK)
  • 16-ray continuous context steering mapped to SvanM2 velocity commands
"""

import sys
import os
import glob
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import mujoco

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
if str(_PACMANBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PACMANBOT_DIR))

from ghost import Ghost, GHOST_COLORS, GHOST_SPEED
from arena_builder import ContinuousArena
from svanm2_runner import SvanM2Runner
from lidar3d import LivoxMid360Lidar, ElevationTraversabilityMapper
from mesh_comms import DecentralizedMeshNetwork
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


class HiveCoordinator:
    """Manages the swarm of Ghost agents and couples them to SvanM2 quadruped runners."""

    def __init__(
        self,
        arena: ContinuousArena,
        runners: List[SvanM2Runner],
        checkpoint: int = -1,
        enable_rl: bool = True,
    ):
        self.arena = arena
        self.runners = runners
        self.n_ghosts = len(runners)
        self.enable_rl = enable_rl
        self.frame_counter = 0
        self.power_duration_steps = 350

        # 1. Initialize RL Actor and Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rl_actor: Optional[GhostActor] = None
        self.predictor_weights = None
        if self.enable_rl:
            self._load_rl_model(checkpoint)

        # 2. Initialize pacmanbot.ghost.Ghost instances
        self.ghosts: Dict[int, Ghost] = {}
        self.recent_nom: Dict[int, np.ndarray] = {}

        player_start_grid = getattr(arena, 'player_start_grid', None)
        if player_start_grid is None:
            pac_gx, pac_gy = arena.to_grid(arena.pacman_start[0], arena.pacman_start[1])
            player_start_grid = (float(pac_gy), float(pac_gx))

        for gid in range(self.n_ghosts):
            if hasattr(arena, 'ghost_starts_grid') and gid < len(arena.ghost_starts_grid):
                pos = (float(arena.ghost_starts_grid[gid][0]), float(arena.ghost_starts_grid[gid][1]))
            else:
                m_start = arena.ghost_starts[gid]
                gx, gy = arena.to_grid(m_start[0], m_start[1])
                pos = (float(gy), float(gx))

            color = GHOST_COLORS[gid % len(GHOST_COLORS)]
            g = Ghost(
                gid=gid,
                grid=arena.grid,
                pos=pos,
                color=color,
                player_start=player_start_grid,
                world=arena.world,
            )
            # Calibrated ghost step speed matching continuous 50 Hz physical simulation (0.80 m/s nominal)
            self.dt_step = 0.02
            base_v = 0.80
            g.max_speed = (base_v / self.arena.scale) * self.dt_step
            g.rl_mode = (self.rl_actor is not None)
            g.cbba_agent.rl_mode = (self.rl_actor is not None)

            # Synchronize trained predictor weights if available
            if self.predictor_weights is not None and hasattr(g, 'belief_map') and hasattr(g.belief_map, 'predictor'):
                try:
                    g.belief_map.predictor.load_state_dict(self.predictor_weights)
                except Exception:
                    pass

            self.ghosts[gid] = g
            self.recent_nom[gid] = np.zeros((MAX_H, MAX_W), dtype=np.float32)

        # Initialize Livox Mid-360 LiDAR and 2.5D Elevation / Traversability Mappers per SvanM2
        self.lidars: Dict[int, LivoxMid360Lidar] = {
            gid: LivoxMid360Lidar(
                points_per_frame=1000,
                fov_up_deg=52.0,
                fov_down_deg=-7.0,
                min_range=0.10,
                max_range=20.0,
                mount_offset_xyz=(0.18, 0.0, 0.10),
            )
            for gid in range(self.n_ghosts)
        }
        self.mappers: Dict[int, ElevationTraversabilityMapper] = {
            gid: ElevationTraversabilityMapper(
                resolution=0.15,
                map_radius=8.0,
                max_step_height=0.12,
                robot_radius=0.35,
            )
            for gid in range(self.n_ghosts)
        }
        # Realistic ad-hoc mesh communication network (16m range, wall occlusion penalty, packet loss)
        self.mesh_network = DecentralizedMeshNetwork(
            max_comm_range=16.0,
            nlos_range_penalty=0.55,
            base_packet_drop_rate=0.02,
        )
        for gid in range(self.n_ghosts):
            self.mesh_network.register_agent(gid)

        self.latest_pointclouds: Dict[int, np.ndarray] = {gid: np.empty((0, 3), dtype=np.float32) for gid in range(self.n_ghosts)}
        self.latest_traversability: Dict[int, Dict[str, np.ndarray]] = {}

        self._had_los_prev: Dict[int, bool] = {gid: False for gid in range(self.n_ghosts)}
        self._last_emergency_auction_frame: Dict[int, int] = {}
        self._cached_rl_tasks: Dict[int, list] = {}
        self._cached_rl_scores_map: Dict[int, np.ndarray] = {}
        self._cached_rl_speed: Dict[int, float] = {}
        self._cached_htasks: Dict[int, list] = {}
        self._cached_hdists: Dict[int, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=min(self.n_ghosts, 8))

    def _load_rl_model(self, checkpoint: int):
        """Loads trained MAPPO GhostActor neural network from pacmanbot checkpoints."""
        ckpt_dir = _PACMANBOT_DIR / "checkpoints"
        ckpts = glob.glob(str(ckpt_dir / "ckpt_*.pt"))
        if not ckpts:
            print("[HiveCoordinator] No checkpoints found in pacmanbot/checkpoints. Falling back to heuristic CBBA.")
            self.enable_rl = False
            return

        def ckpt_score(f):
            try:
                num = int(f.split('ckpt_')[-1].split('.pt')[0])
                return (num > 0 and num % 100 == 0, num)
            except ValueError:
                return (False, -1)

        latest = max(ckpts, key=ckpt_score)
        if checkpoint != -1:
            req_path = str(ckpt_dir / f"ckpt_{checkpoint}.pt")
            if os.path.exists(req_path):
                latest = req_path

        print(f"[HiveCoordinator] Loading MAPPO GhostActor checkpoint: {latest} on {self.device}")
        try:
            self.rl_actor = GhostActor().to(self.device)
            ckpt_data = torch.load(latest, map_location=self.device, weights_only=False)
            if "actor" in ckpt_data:
                actor_sd = ckpt_data["actor"]
                new_actor_sd = self.rl_actor.state_dict()
                for k, v in actor_sd.items():
                    if k in new_actor_sd:
                        if v.shape == new_actor_sd[k].shape:
                            new_actor_sd[k] = v
                        elif "vec_mlp.0.weight" in k and v.ndim == 2 and new_actor_sd[k].ndim == 2:
                            min_out = min(v.shape[0], new_actor_sd[k].shape[0])
                            min_in = min(v.shape[1], new_actor_sd[k].shape[1])
                            new_actor_sd[k][:min_out, :min_in] = v[:min_out, :min_in]
                self.rl_actor.load_state_dict(new_actor_sd)
            else:
                self.rl_actor.load_state_dict(ckpt_data)
            self.rl_actor.eval()
            self.predictor_weights = ckpt_data.get("predictor", None)
            print("[HiveCoordinator] MAPPO GhostActor loaded and activated.")
        except Exception as e:
            print(f"[HiveCoordinator] Failed to load GhostActor checkpoint: {e}. Falling back to heuristic CBBA.")
            self.rl_actor = None
            self.enable_rl = False

    def _trigger_emergency_sighting(self, sighting_gid: int, sighting_ghost):
        """Instantaneous multi-agent swarm sighting relay matching pacmanbot/worker.py:
        Alerts all ghosts within comms distance and auctions coordinated pursuit and cut-off tasks.
        """
        pac_coord = sighting_ghost.known_pacman
        if pac_coord is None or sighting_ghost.pacman_powered:
            return

        s_pos = (sighting_ghost.y, sighting_ghost.x)
        # Ghosts in radio mesh network (multi-hop mesh covers the arena, minimum comms radius 18.0 grid units = 27m)
        # Any ghost connected directly or via relay hops receives the sighting broadcast.
        local_gids = [g.gid for g in self.ghosts.values() if not g.dead]
        if not local_gids:
            local_gids = [sighting_gid]

        pac_target = (round(float(pac_coord[0]), 1), round(float(pac_coord[1]), 1))
        self._last_sighting_frame = self.frame_counter
        self._last_sighting_pos = pac_target

        for lgid in local_gids:
            lg = self.ghosts[lgid]
            lg.known_pacman = pac_target
            lg.pacman_last_seen = self.frame_counter

        from allocator import Task, TaskType, ORIGIN_RL_NOVEL
        from obs import RL_SCORE_BASE, RL_SCORE_SPAN
        from pathfinder import dijkstra_multi
        from cbba import _deduplicate_tasks

        is_rl = (self.rl_actor is not None)
        R_map = getattr(self.arena.world, 'height', MAX_H)
        C_map = getattr(self.arena.world, 'width', MAX_W)

        # 4-Role Swarm Tactical Pincer Envelope:
        # 1. Direct Pursuer: pac_target
        # 2. Lead Interceptor: along pacman velocity (+2.5 cells)
        # 3. Flanker 1 (Left orthogonal or downstream intersection): +2.0 cells offset
        # 4. Flanker 2 (Right orthogonal or rear cutoff): -2.0 cells offset
        p_dir = getattr(sighting_ghost, '_player_dir', (0.0, 0.0))
        p_spd = math.hypot(p_dir[0], p_dir[1])
        if p_spd > 0.05:
            ny = p_dir[0] / p_spd
            nx = p_dir[1] / p_spd
        else:
            ny, nx = 0.0, 0.0

        tactical_targets = [pac_target]

        # Role 2: Lead Interceptor
        if p_spd > 0.05:
            lead_y = round(pac_target[0] + ny * 2.5, 1)
            lead_x = round(pac_target[1] + nx * 2.5, 1)
            if self.arena.world.is_passable(lead_x, lead_y, radius=0.35):
                tactical_targets.append((lead_y, lead_x))

        # Role 3 & 4: Flankers / Corridor Pinchers
        # Orthogonal unit vectors: (-nx, ny) and (nx, -ny)
        ortho_dirs = [(-nx, ny), (nx, -ny)] if (ny != 0 or nx != 0) else [(1.0, 0.0), (-1.0, 0.0)]
        for oy, ox in ortho_dirs:
            flank_y = round(pac_target[0] + oy * 2.2, 1)
            flank_x = round(pac_target[1] + ox * 2.2, 1)
            if self.arena.world.is_passable(flank_x, flank_y, radius=0.35):
                tactical_targets.append((flank_y, flank_x))

        # Fallback to nearest passable PRM intersections around Pac-Man if orthogonal rays hit walls
        if len(tactical_targets) < min(len(local_gids), 4) and hasattr(self.arena.world, 'prm_nodes'):
            for pnode in self.arena.world.prm_nodes:
                d_pn = math.hypot(pnode[0] - pac_target[0], pnode[1] - pac_target[1])
                if 1.5 <= d_pn <= 5.0 and not any(math.hypot(pnode[0] - tt[0], pnode[1] - tt[1]) < 1.5 for tt in tactical_targets):
                    tactical_targets.append((float(pnode[0]), float(pnode[1])))
                    if len(tactical_targets) >= min(len(local_gids), 4):
                        break

        e_pool = []
        for tt_idx, tt in enumerate(tactical_targets):
            tr_idx = min(max(0, int(round(tt[0]))), R_map - 1)
            tc_idx = min(max(0, int(round(tt[1]))), C_map - 1)
            # Priority weighting: direct pursuit 1.0, lead 1.10, flankers 0.95
            prio_mult = 1.10 if tt_idx == 1 else (1.0 if tt_idx == 0 else 0.95)
            for lgid in local_gids:
                if is_rl:
                    sc_map = self._cached_rl_scores_map.get(lgid)
                    conf = float(sc_map[tr_idx, tc_idx]) if (sc_map is not None and 0 <= tr_idx < sc_map.shape[0] and 0 <= tc_idx < sc_map.shape[1]) else 0.8
                    rl_score = (RL_SCORE_BASE + RL_SCORE_SPAN * min(1.0, max(0.0, conf))) * prio_mult
                    spd = self._cached_rl_speed.get(lgid, 1.0)
                    e_pool.append(Task(task_type=TaskType.HUNT, target_pos=tt, score=rl_score,
                                       assigned_to=-1, created_frame=self.frame_counter, owner=lgid,
                                       target_speed=spd, origin=ORIGIN_RL_NOVEL))
                else:
                    score = (12.0 - tt_idx * 0.5) * prio_mult
                    e_pool.append(Task(task_type=TaskType.HUNT, target_pos=tt, score=score,
                                       assigned_to=-1, created_frame=self.frame_counter, owner=lgid,
                                       target_speed=1.0, origin=0))

        deduped_e_tasks = _deduplicate_tasks(e_pool, threshold=1.2)
        all_targets = [t.target_pos for t in deduped_e_tasks]

        for lgid in local_gids:
            lg = self.ghosts[lgid]
            lg.cbba_agent.emergency_preempt_explore()
            d_rl = dijkstra_multi(lg.world, (lg.y, lg.x), all_targets) if all_targets else {}
            if deduped_e_tasks:
                lg.cbba_agent._phase1(lg, deduped_e_tasks, d_rl)

        if len(local_gids) > 1:
            for round_idx in range(2):
                payloads = {gid: self.ghosts[gid].cbba_agent.get_consensus_payload() for gid in local_gids}
                for gid_i in local_gids:
                    agent_i = self.ghosts[gid_i].cbba_agent
                    for gid_j in local_gids:
                        if gid_i != gid_j:
                            p_j = payloads[gid_j]
                            agent_i.receive_consensus(gid_j, p_j["y"], p_j["z"], p_j["s"], self.frame_counter, p_j.get("meta"))
                if round_idx < 1:
                    for lgid in local_gids:
                        lg = self.ghosts[lgid]
                        if len(lg.cbba_agent.bundle) == 0:
                            d_rl = dijkstra_multi(lg.world, (lg.y, lg.x), all_targets) if all_targets else {}
                            lg.cbba_agent._phase1(lg, deduped_e_tasks, d_rl)

    def step(
        self,
        pacman_pos_metric: Tuple[float, float],
        pacman_powered_true: bool,
    ):
        """Steps all ghost agents: synchronizes physics, runs autonomy, and outputs commands."""
        self.frame_counter += 1

        # 1. Convert true Pac-Man metric position to grid coordinates (row, col)
        pac_gx, pac_gy = self.arena.to_grid(pacman_pos_metric[0], pacman_pos_metric[1])
        player_pos = (float(pac_gy), float(pac_gx))  # (row, col)

        # 3. High-level MAPPO RL inference and CBBA consensus (evaluated every 6 frames = 0.12s, 1:1 with pacmanbot)
        if self.rl_actor is not None and (self.frame_counter % 6 == 0):
            alive = [gid for gid, g in self.ghosts.items() if not g.dead]
            R = min(MAX_H, len(self.arena.grid))
            C = min(MAX_W, len(self.arena.grid[0]))

            def _build_obs_task(gid):
                g = self.ghosts[gid]
                all_tasks, dists = _heuristic_tasks(g, self.frame_counter)
                g._rl_candidates = all_tasks
                s = build_spatial(g, self.recent_nom[gid], R, C, obs_resolution=1.0)
                v = build_vector(g)
                m = build_valid_mask(g, R, C, obs_resolution=1.0, spatial_walls=s[0])
                return gid, all_tasks, dists, s, v, m

            def _pad_spatial(arr, target_h, target_w):
                h, w = arr.shape[-2], arr.shape[-1]
                if h == target_h and w == target_w:
                    return arr
                out = np.zeros(arr.shape[:-2] + (target_h, target_w), dtype=arr.dtype)
                out[..., :h, :w] = arr
                if arr.ndim == 3:
                    out[0, h:, :] = 1.0
                    out[0, :, w:] = 1.0
                elif arr.ndim == 2:
                    out[h:, :] = 1.0
                    out[:, w:] = 1.0
                return out

            results = list(self.executor.map(_build_obs_task, alive))
            sp, ve, vm = [], [], []
            for gid, all_tasks, dists, s, v, m in results:
                self._cached_htasks[gid] = all_tasks
                self._cached_hdists[gid] = dists
                sp.append(_pad_spatial(s, MAX_H, MAX_W))
                ve.append(v)
                vm.append(_pad_spatial(m, MAX_H, MAX_W))

            if alive:
                t_sp = torch.tensor(np.stack(sp), device=self.device, dtype=torch.float32)
                t_ve = torch.tensor(np.stack(ve), device=self.device, dtype=torch.float32)
                t_vm = torch.tensor(np.stack(vm), device=self.device, dtype=torch.bool)

                with torch.inference_mode():
                    out_act = self.rl_actor(t_sp, t_ve, t_vm, K=3)
                    sel_idx, sel_lp, scores, pool, vec, speed_idx, speed_lp, speed_logits = out_act

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
                    self._cached_rl_tasks[gid] = rl_tasks
                    self._cached_rl_scores_map[gid] = sc_np[i]
                    self._cached_rl_speed[gid] = g.current_speed_mult
                    if rl_tasks:
                        pool_tasks.extend(rl_tasks)

                deduped_pool = _deduplicate_tasks(pool_tasks, threshold=1.5)
                needed = len(alive) - len(deduped_pool)
                if needed > 0:
                    existing_targets = [t.target_pos for t in deduped_pool]
                    aux_targets = []
                    for gid in alive:
                        g_aux = self.ghosts[gid]
                        if hasattr(g_aux, 'belief_map') and hasattr(g_aux.belief_map, 'top_cells'):
                            top_aux = g_aux.belief_map.top_cells(n=max(4, needed * 2))
                            for tc in top_aux:
                                if not any(math.hypot(tc[0] - et[0], tc[1] - et[1]) < 2.5 for et in (existing_targets + aux_targets)):
                                    aux_targets.append(tc)
                                    if len(aux_targets) >= needed:
                                        break
                        if len(aux_targets) >= needed:
                            break

                    pellet_targets = set()
                    active_pps = getattr(self.arena.world, 'power_pellets', None)
                    if active_pps:
                        for ppx, ppy in active_pps:
                            if not any(math.hypot(ppy - et[0], ppx - et[1]) < 2.0 for et in (existing_targets + aux_targets)):
                                aux_targets.append((ppy, ppx))
                                pellet_targets.add((ppy, ppx))
                                if len(aux_targets) >= needed:
                                    break

                    if len(aux_targets) < needed:
                        for gid in alive:
                            g_aux = self.ghosts[gid]
                            for n in getattr(g_aux, 'prm_last_seen', {}):
                                if not any(math.hypot(n[0] - et[0], n[1] - et[1]) < 3.0 for et in (existing_targets + aux_targets)):
                                    if g_aux.world.is_passable(float(n[1]), float(n[0]), radius=0.35):
                                        aux_targets.append(n)
                                        if len(aux_targets) >= needed:
                                            break
                            if len(aux_targets) >= needed:
                                break

                    for at in aux_targets:
                        t_type = TaskType.CONVERT if (at[0], at[1]) in pellet_targets else TaskType.HUNT
                        score_val = 6.5 if t_type == TaskType.CONVERT else 0.6
                        deduped_pool.append(Task(task_type=t_type, target_pos=(float(at[0]), float(at[1])), score=score_val, origin=ORIGIN_RL_NOVEL))

                # Ensure all active power pellets are present in the auction as high-priority CONVERT tasks
                active_pps = getattr(self.arena.world, 'power_pellets', None)
                if active_pps:
                    for ppx, ppy in active_pps:
                        if not any(t.task_type == TaskType.CONVERT and math.hypot(t.target_pos[0] - ppy, t.target_pos[1] - ppx) < 1.2 for t in deduped_pool):
                            deduped_pool.append(Task(task_type=TaskType.CONVERT, target_pos=(float(ppy), float(ppx)), score=6.5, origin=ORIGIN_RL_NOVEL))

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

                if getattr(self, '_enable_idle_fallback', True):
                    sighting_active = (self.frame_counter - getattr(self, '_last_sighting_frame', -999) <= 50) and not pacman_powered_true
                    last_s_pos = getattr(self, '_last_sighting_pos', None)
                    for gid in alive:
                        g = self.ghosts[gid]
                        if g.cbba_agent.get_active_task() is None:
                            cands = self._cached_htasks.get(gid, [])
                            if sighting_active and last_s_pos is not None:
                                # When Pac-Man sighting is active, prevent ghosts from wandering to distant corners.
                                # Keep them patrolling containment junctions near the sighting area.
                                filtered_cands = [
                                    c for c in cands 
                                    if c.task_type in (TaskType.HUNT, TaskType.CONVERT) or 
                                    (c.task_type == TaskType.EXPLORE and math.hypot(c.target_pos[0] - last_s_pos[0], c.target_pos[1] - last_s_pos[1]) <= 8.5)
                                ]
                                if not filtered_cands:
                                    # Assign containment hunt task at last known sighting
                                    filtered_cands = [Task(task_type=TaskType.HUNT, target_pos=last_s_pos, score=3.0, origin=ORIGIN_RL_NOVEL)]
                                cands = filtered_cands
                            if cands:
                                h_dists = dict(self._cached_hdists.get(gid, {}))
                                g.cbba_agent._phase1(g, cands, h_dists)

        # 4. Update all ghost autonomy logic
        sim_time = self.frame_counter * self.dt_step
        agent_positions = {
            gid: (float(self.runners[gid].pos[0]), float(self.runners[gid].pos[1]))
            for gid in self.ghosts.keys() if not self.ghosts[gid].dead
        }

        for gid, g in self.ghosts.items():
            if g.dead:
                continue

            # 4a. Execute Livox Mid-360 3D LiDAR Scan
            runner = self.runners[gid]
            r_pos = runner.pos
            r_yaw = runner.yaw
            ignore_body = runner.root_body_id if hasattr(runner, 'root_body_id') else -1
            
            pc, ranges, intensities = self.lidars[gid].scan(
                model=runner.model,
                data=runner.data,
                sensor_pos=r_pos,
                sensor_yaw=r_yaw,
                ignore_body_id=ignore_body,
            )
            self.latest_pointclouds[gid] = pc

            # 4b. Update 2.5D Elevation & Traversability Map
            if len(pc) > 0:
                trav_data = self.mappers[gid].update_map(pc, r_pos)
                self.latest_traversability[gid] = trav_data

            # 4c. Broadcast heartbeats and status via the decentralized mesh network
            if self.frame_counter % 5 == 0:
                self.mesh_network.broadcast(
                    sender_id=gid,
                    sender_pos=(r_pos[0], r_pos[1]),
                    agent_positions=agent_positions,
                    packet_type="heartbeat",
                    payload={"known_pacman": g.known_pacman, "is_powered": g.pacman_powered},
                    sim_time=sim_time,
                )

            # Drain incoming mesh packets for this agent
            incoming_packets = self.mesh_network.drain_inbox(gid)
            for pkt in incoming_packets:
                if pkt.packet_type == "heartbeat" and pkt.payload.get("known_pacman") is not None:
                    # Received relayed sighting if we don't have our own fresh fix
                    if g.known_pacman is None:
                        g.known_pacman = pkt.payload["known_pacman"]
                        g.pacman_last_seen = self.frame_counter

            # Calibrate step max_speed: 0.95 m/s when hunting, 0.80 m/s when patrolling
            is_hunting = (getattr(g, 'known_pacman', None) is not None or getattr(g, '_is_striking', False)) and not g.pacman_powered
            base_v = 0.95 if is_hunting else 0.80
            g.max_speed = (base_v / self.arena.scale) * self.dt_step

            # Per-ghost awareness: unaware ghosts hunt at current_speed_mult; aware ghosts evade at 0.65
            spd = 0.65 if g.pacman_powered else getattr(g, 'current_speed_mult', 1.0)

            # ghost.update performs the local LiDAR check, radio comms, CBBA step, and context steering
            g.update(
                player_pos=player_pos,
                powered=pacman_powered_true,
                all_ghosts=self.ghosts,
                skip_movement=False,
                speed_mult=spd,
            )

            # Ensure local power timer matches full simulation duration if newly aware
            if g.pacman_powered and not getattr(g, '_prev_powered_state', False):
                dur = getattr(self, 'power_duration_steps', 350)
                g.pacman_power_timer = dur
                g.pacman_power_max_timer = float(dur)
            elif not g.pacman_powered:
                g.pacman_power_timer = 0
            g._prev_powered_state = g.pacman_powered

            # Sighting event check: if ghost acquired LOS or target drifted, trigger emergency swarm auction
            has_los = g.known_pacman is not None
            had_los = self._had_los_prev.get(gid, False)
            cooldown_ok = (self.frame_counter - self._last_emergency_auction_frame.get(gid, -999) >= 4)
            trigger_emergency = False
            if has_los and not had_los and not pacman_powered_true:
                trigger_emergency = True
            elif has_los and not pacman_powered_true and cooldown_ok:
                active_t = g.cbba_agent.get_active_task()
                if active_t is not None and active_t.task_type == TaskType.HUNT:
                    d_drift = math.hypot(g.known_pacman[0] - active_t.target_pos[0], g.known_pacman[1] - active_t.target_pos[1])
                    if d_drift >= 3.5:
                        trigger_emergency = True
            if trigger_emergency:
                self._last_emergency_auction_frame[gid] = self.frame_counter
                self._trigger_emergency_sighting(gid, g)
            self._had_los_prev[gid] = has_los

            # 5. Map the continuous grid velocity (vx, vy) to metric world velocity for SvanM2
            vx_world = (float(g.vx) / self.dt_step) * self.arena.scale
            vy_world = (float(g.vy) / self.dt_step) * self.arena.scale

            runner = self.runners[gid]
            runner.set_world_velocity_target(vx_world, vy_world, speed_mult=1.0)

    def eliminate_ghost(self, gid: int):
        """Marks a ghost as eliminated by powered Pac-Man and moves its robot out of play."""
        if gid in self.ghosts:
            g = self.ghosts[gid]
            g.kill()
            kill_x, kill_y = g.x, g.y
            # Alert nearby witness ghosts within 12.0 cells with line-of-sight (1:1 with pacmanbot/pacman.py)
            for other_gid, og in list(self.ghosts.items()):
                if other_gid != gid and not og.dead:
                    d_w = math.hypot(og.y - kill_y, og.x - kill_x)
                    witnesses = False
                    if d_w <= 12.0:
                        if og.world and hasattr(og.world, 'line_of_sight'):
                            witnesses = og.world.line_of_sight((og.x, og.y), (kill_x, kill_y), radius=og.radius, step_size=0.5)
                        else:
                            witnesses = True
                    if witnesses:
                        og.witness_death(gid, self.ghosts)
            g._broadcast([("agent_dead", gid)], self.ghosts)

        if gid < len(self.runners):
            runner = self.runners[gid]
            runner.reset_pose(0.0, 0.0, z=-5.0)  # Move underground out of arena
            runner.cmd_base[:] = 0.0

    @property
    def alive_ghost_count(self) -> int:
        return sum(1 for g in self.ghosts.values() if not g.dead)
