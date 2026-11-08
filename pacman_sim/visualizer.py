#!/usr/bin/env python3
"""
visualizer.py — High-Performance 2D Gamestate Visualizer for MuJoCo Pac-Man.

Faithfully matches the continuous visualizer from pacmanbot/pacman.py:
  • Left Panel: Continuous Global Game Arena (continuous capsule/curved wall obstacles,
    pellets, power pellets, autonomous 4-mecanum rover car, and authentic ghosts with
    animated skirts, wavy cutouts, directional eyes).
  • Right Panel: Decentralized Ghost Swarm Mind (Fog-of-war personal map, 12m NRF24 radio
    network known agents, particle filter Bayesian belief heatmap, keys [0-6] to switch).
  • Bottom HUD: Live score, power timer progress bar, AUTO MODE / MANUAL PLAY button,
    RL MODE button, and game-over / victory banners.

Usage:
  python pacman_sim/visualizer.py --stage 1
  python pacman_sim/visualizer.py --stage 2 --speedup 2.0
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Set

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import numpy as np

# Ensure repository paths are present
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"

for p in [str(_PACMANBOT_DIR), str(_PACMAN_SIM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from game_engine import PacmanGameEngine
from arena_builder import GHOST_COLORS
from pacman import (
    CELL, WIDTH, HEIGHT, FPS,
    BLACK, WHITE, YELLOW, BLUE, RED, PINK, CYAN, ORANGE, DKBLUE, GREY,
    POWERED_COLOR, PURPLE, GHOST_COLORS as PAC_GHOST_COLORS,
    WALL, EMPTY, PELLET, POWER,
    TOGGLE_WIDTH, TOGGLE_HEIGHT, draw_mecanum_car
)
from ghost import Ghost, UNKNOWN, RADIUS as _COMM_RADIUS


class Pacman2DVisualizer:
    """Continuous dual-panel Pygame visualizer based directly on pacmanbot/pacman.py."""

    def __init__(
        self,
        stage: int = 1,
        seed: Optional[int] = None,
        speedup: float = 1.0,
        engine: Optional[PacmanGameEngine] = None,
        speed_state: Optional[Dict[str, Any]] = None,
    ):
        self.stage = stage
        self.seed = seed
        self.speedup = speedup
        self.paused = False
        self.debug_ghost_id = 0
        self.auto_mode = True
        self.rl_mode = True
        self.speed_state = speed_state
        if self.speed_state is not None:
            self.speedup = self.speed_state.get("speedup", speedup)
            self.paused = self.speed_state.get("paused", False)

        # Initialize game engine (reuse existing if supplied)
        if engine is not None:
            self.engine = engine
        else:
            print(f"[Visualizer] Initializing MuJoCo Game Engine for Stage {stage}...")
            self.engine = PacmanGameEngine(stage=stage, seed=seed, scale=1.5)

        self.arena = self.engine.arena
        self.world = self.arena.world
        self.grid = self.arena.grid
        self.player = self.engine.pacman
        self.ghosts = self.engine.hive.ghosts
        self.cols = self.arena.cols
        self.rows = self.arena.rows
        self.n_ghosts = self.arena.n_ghosts

        # Authentic cell scaling matching pacmanbot/pacman.py (CELL = 20)
        self.cell_size = CELL
        self.width = self.cols * self.cell_size
        self.hud_height = 48
        self.height = self.rows * self.cell_size + self.hud_height

        # Initialize Pygame window with dual-panel layout matching pacman.py
        pygame.init()
        try:
            pygame.font.init()
            self.font  = pygame.font.SysFont("monospace", 18, bold=True)
            self.small = pygame.font.SysFont("monospace", 14)
        except Exception:
            self.font  = pygame.font.Font(None, 22)
            self.small = pygame.font.Font(None, 16)

        pygame.display.set_caption(f"PACMAN (Continuous MuJoCo) — Stage {self.stage}")
        self.screen = pygame.display.set_mode((self.width * 2, self.height))
        self.clock = pygame.time.Clock()

        # Pre-render continuous obstacle background surface matching pacman.py:draw_grid()
        self._bg_surface = None
        self._prerender_background()

        # Dynamic responsive button positioning for any curriculum stage
        btn_w = min(160, max(80, (self.width - 130) // 2))
        btn_h = 32
        btn_y = self.rows * self.cell_size + 8
        rl_x = self.width - btn_w - 10
        auto_x = rl_x - btn_w - 10
        self.toggle_rect = pygame.Rect(max(100, auto_x), btn_y, btn_w, btn_h)
        self.rl_toggle_rect = pygame.Rect(max(190, rl_x), btn_y, btn_w, btn_h)

        # Caches for personal map
        self._prm_cache = None
        self._prm_cache_n = 0
        self._heat_cache: Dict[int, Tuple[pygame.Surface, int]] = {}

        self.sim_time = 0.0
        self.step_dt = 0.020  # 50 Hz

    def _prerender_background(self):
        """Bakes continuous obstacle geometry (capsules and curved walls) onto _bg_surface."""
        self._bg_surface = pygame.Surface((self.width, self.rows * self.cell_size))
        self._bg_surface.fill(BLACK)
        if hasattr(self.world, "obstacles") and self.world.obstacles:
            for obs in self.world.obstacles:
                obs.draw(self._bg_surface, self.cell_size)

    def draw_grid(self):
        """Draws continuous arena background and live pellets matching pacman.py:L1161-L1178."""
        if self._bg_surface is None:
            self._prerender_background()
        self.screen.blit(self._bg_surface, (0, 0))

        # Draw live regular pellets directly from continuous world model
        for px, py in self.world.pellets:
            x = int(px * self.cell_size)
            y = int(py * self.cell_size)
            pygame.draw.circle(self.screen, WHITE, (x, y), 2)

        # Draw live power pellets directly from continuous world model
        for px, py in self.world.power_pellets:
            x = int(px * self.cell_size)
            y = int(py * self.cell_size)
            pygame.draw.circle(self.screen, WHITE, (x, y), 5)

    def draw_hud(self):
        """Draws status console: Score, Powered bar, AUTO MODE toggle, RL MODE toggle matching pacman.py:L1179-L1202."""
        y = self.rows * self.cell_size
        pygame.draw.rect(self.screen, BLACK, (0, y, self.width * 2, self.hud_height))

        # Score
        score_txt = self.font.render(f"SCORE  {self.player.score}", True, WHITE)
        self.screen.blit(score_txt, (10, y + 6))

        # Powered state progress bar
        if self.player.powered:
            max_dur = getattr(self.engine, 'power_duration_steps', 40)
            cur_timer = getattr(self.player, 'power_timer', 0)
            bar_w = int((cur_timer / max(1, max_dur)) * 100)
            pygame.draw.rect(self.screen, GREY, (self.width // 2 - 50, y + 28, 100, 8))
            pygame.draw.rect(self.screen, POWERED_COLOR, (self.width // 2 - 50, y + 28, max(0, bar_w), 8))
            txt = self.small.render("POWERED", True, POWERED_COLOR)
            self.screen.blit(txt, (self.width // 2 - 28, y + 10))

        # Auto Mode / Manual Mode Button
        bg_btn = (0, 200, 100) if self.auto_mode else GREY
        pygame.draw.rect(self.screen, bg_btn, self.toggle_rect, border_radius=4)
        lbl_msg = "AUTO MODE" if self.auto_mode else "MANUAL PLAY"
        text_btn = self.small.render(lbl_msg, True, WHITE)
        text_rect = text_btn.get_rect(center=self.toggle_rect.center)
        self.screen.blit(text_btn, text_rect)

        # RL Mode Button
        bg_btn_rl = (200, 0, 100) if self.rl_mode else GREY
        pygame.draw.rect(self.screen, bg_btn_rl, self.rl_toggle_rect, border_radius=4)
        lbl_msg_rl = "RL MODE ON" if self.rl_mode else "RL MODE OFF"
        text_btn_rl = self.small.render(lbl_msg_rl, True, WHITE)
        text_rect_rl = text_btn_rl.get_rect(center=self.rl_toggle_rect.center)
        self.screen.blit(text_btn_rl, text_rect_rl)

    def draw_personal_map(self):
        """Draws Ghost Swarm Personal Map matching pacman.py:L1203-L1301."""
        ghost = self.ghosts.get(self.debug_ghost_id)
        if not ghost:
            if self.ghosts:
                ghost = next(iter(self.ghosts.values()))
            else:
                return

        ox = self.width
        old_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(ox, 0, self.width, self.rows * self.cell_size))
        pygame.draw.rect(self.screen, BLACK, (ox, 0, self.width, self.rows * self.cell_size))

        # 1. Incremental lavender LiDAR boundary points
        if hasattr(ghost, "lidar_memory"):
            if not hasattr(ghost, "_lidar_surf") or ghost._lidar_surf is None:
                ghost._lidar_surf = pygame.Surface((self.width, self.rows * self.cell_size), pygame.SRCALPHA)
                ghost._lidar_drawn = set()
            new_pts = ghost.lidar_memory - getattr(ghost, "_lidar_drawn", set())
            if new_pts:
                for r, c in new_pts:
                    pygame.draw.rect(
                        ghost._lidar_surf,
                        PURPLE,
                        (int(c * self.cell_size - 2), int(r * self.cell_size - 2), 4, 4),
                    )
                ghost._lidar_drawn.update(new_pts)
            self.screen.blit(ghost._lidar_surf, (ox, 0))

        # 2. Known regular and power pellets
        if hasattr(ghost, "known_pellets"):
            for px, py in ghost.known_pellets:
                pygame.draw.circle(
                    self.screen, WHITE, (int(ox + px * self.cell_size), int(py * self.cell_size)), 2
                )
        if hasattr(ghost, "known_power_pellets"):
            for px, py in ghost.known_power_pellets:
                pygame.draw.circle(
                    self.screen, WHITE, (int(ox + px * self.cell_size), int(py * self.cell_size)), 5
                )

        # 3. Navigable PRM graph nodes
        bm = getattr(ghost, "belief_map", None)
        if bm is not None and getattr(bm, "_open_cells", None):
            if not hasattr(self, "_prm_cache") or self._prm_cache is None or getattr(self, "_prm_cache_n", 0) != bm.n_nodes:
                self._prm_cache = pygame.Surface((self.width, self.rows * self.cell_size), pygame.SRCALPHA)
                for n in bm._open_cells:
                    pygame.draw.rect(
                        self._prm_cache, (30, 30, 30), (int(n[1] * self.cell_size), int(n[0] * self.cell_size), 4, 4)
                    )
                self._prm_cache_n = bm.n_nodes
            self.screen.blit(self._prm_cache, (ox, 0))

        # 4. Bayesian Belief Heatmap
        if bm is not None and getattr(bm, "_initialised", False) and len(getattr(bm, "_b_flat", [])) > 0:
            active_mask = bm._b_flat >= 0.001
            if np.any(active_mask):
                active_indices = np.where(active_mask)[0]
                max_p = float(np.max(bm._b_flat[active_indices]))
                if max_p > 1e-9:
                    if len(active_indices) > 80:
                        top_sub = np.argpartition(bm._b_flat[active_indices], -80)[-80:]
                        active_indices = active_indices[top_sub]
                    for idx in active_indices:
                        r, c = bm._open_cells[idx]
                        p = float(bm._b_flat[idx])
                        t = min(1.0, p / max_p)
                        t_idx = int(t * 31)
                        if t_idx not in self._heat_cache:
                            t_d = t_idx / 31.0
                            red = int(t_d * 255)
                            green = int((1.0 - t_d) * 40)
                            blue = int((1.0 - t_d) * 210)
                            alpha = int(40 + t_d * 140)
                            s_r = int(3 + t_d * 4)
                            surf = pygame.Surface((s_r * 2, s_r * 2), pygame.SRCALPHA)
                            pygame.draw.circle(surf, (red, green, blue, alpha), (s_r, s_r), s_r)
                            self._heat_cache[t_idx] = (surf, s_r)
                        surf, s_r = self._heat_cache[t_idx]
                        self.screen.blit(surf, (ox + int(c * self.cell_size) - s_r, int(r * self.cell_size) - s_r))

        # 5. CBBA active task target (Yellow circle + crosshair)
        if hasattr(ghost, "cbba_agent"):
            active_task = ghost.cbba_agent.get_active_task()
            if active_task is not None and hasattr(active_task, "target_pos"):
                tr, tc = active_task.target_pos
                tx = ox + int(tc * self.cell_size)
                ty = int(tr * self.cell_size)
                pygame.draw.circle(self.screen, (255, 255, 0), (tx, ty), self.cell_size // 2 + 2, 2)
                pygame.draw.line(self.screen, (255, 255, 0), (tx - 4, ty - 4), (tx + 4, ty + 4), 2)
                pygame.draw.line(self.screen, (255, 255, 0), (tx - 4, ty + 4), (tx + 4, ty - 4), 2)

        # 6. Radio communication radius circle (12m range)
        cx = ox + int(ghost.x * self.cell_size)
        cy = int(ghost.y * self.cell_size)
        pygame.draw.circle(self.screen, (60, 60, 120), (cx, cy), int(_COMM_RADIUS * self.cell_size), 1)

        # 7. Known peer agents from mesh network
        for gid, pos in ghost.known_agents.items():
            if pos == "UNKNOWN" or pos is None:
                continue
            gr, gc = pos
            ax = ox + int(gc * self.cell_size)
            ay = int(gr * self.cell_size)
            c_col = PAC_GHOST_COLORS[gid % len(PAC_GHOST_COLORS)]
            pygame.draw.circle(self.screen, c_col, (ax, ay), self.cell_size // 2 - 2)
            label = self.small.render(str(gid), True, WHITE)
            self.screen.blit(label, (ax - 4, ay - 6))

        # 8. Ghost itself on personal map
        ghost.draw(self.screen, scale=self.cell_size, offset_x=ox)

        # 9. Known Pac-Man
        if ghost.known_pacman:
            pr, pc = ghost.known_pacman
            px = ox + int(pc * self.cell_size)
            py = int(pr * self.cell_size)
            pac_col = POWERED_COLOR if ghost.pacman_powered else YELLOW
            pygame.draw.circle(self.screen, pac_col, (px, py), self.cell_size // 2 - 2)
            label = self.small.render("P", True, BLACK)
            self.screen.blit(label, (px - 4, py - 6))

        self.screen.set_clip(old_clip)

        # Right footer status banner
        mode = "POWERED" if ghost.pacman_powered else ("HUNT" if ghost.known_pacman else "SEARCH")
        fb = " [FALLBACK]" if getattr(ghost, "in_fallback_mode", False) else ""
        txt = self.small.render(f"Ghost {self.debug_ghost_id} [{mode}{fb}]  [0-6 to switch]", True, WHITE)
        self.screen.blit(txt, (ox + 4, self.rows * self.cell_size + 6))

    def draw_overlay(self, msg: str, color=WHITE):
        """Draws centered victory or game-over banner matching pacman.py:L1302-L1309."""
        overlay = pygame.Surface((self.width, self.rows * self.cell_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        txt = self.font.render(msg, True, color)
        rect = txt.get_rect(center=(self.width // 2, (self.rows * self.cell_size) // 2))
        self.screen.blit(txt, rect)

    def handle_events(self) -> bool:
        """Processes keyboard and mouse events matching pacman.py interaction."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    if self.speed_state is not None:
                        self.speed_state["paused"] = self.paused
                    state_str = "PAUSED" if self.paused else "RESUMED"
                    print(f"  >> Simulation {state_str}")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speedup = min(8.0, self.speedup + 0.5)
                    if self.speed_state is not None:
                        self.speed_state["speedup"] = self.speedup
                    print(f"  >> Playback speed: {self.speedup:.1f}x")
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    self.speedup = max(0.5, self.speedup - 0.5)
                    if self.speed_state is not None:
                        self.speed_state["speedup"] = self.speedup
                    print(f"  >> Playback speed: {self.speedup:.1f}x")
                elif event.key in (pygame.K_m, pygame.K_TAB):
                    self.auto_mode = not self.auto_mode
                    self.player.manual_mode = not self.auto_mode
                    self.player.manual_vx = 0.0
                    self.player.manual_vy = 0.0
                    self.player._route = []
                    self.player._route_target = None
                    mode_str = "AUTO MODE" if self.auto_mode else "MANUAL PLAY"
                    print(f"  >> Control Mode: {mode_str}")
                elif event.key in (pygame.K_w, pygame.K_UP):
                    if not self.auto_mode:
                        self.player.manual_mode = True
                        self.player.manual_vy = -1.0
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    if not self.auto_mode:
                        self.player.manual_mode = True
                        self.player.manual_vy = 1.0
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    if not self.auto_mode:
                        self.player.manual_mode = True
                        self.player.manual_vx = -1.0
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    if not self.auto_mode:
                        self.player.manual_mode = True
                        self.player.manual_vx = 1.0
                elif pygame.K_0 <= event.key <= pygame.K_6:
                    num = event.key - pygame.K_0
                    if num in self.ghosts:
                        self.debug_ghost_id = num
                        print(f"  >> Switched personal map view to Ghost {num}")
            elif event.type == pygame.KEYUP:
                if not self.auto_mode:
                    if event.key in (pygame.K_w, pygame.K_UP) and self.player.manual_vy < 0:
                        self.player.manual_vy = 0.0
                    elif event.key in (pygame.K_s, pygame.K_DOWN) and self.player.manual_vy > 0:
                        self.player.manual_vy = 0.0
                    elif event.key in (pygame.K_a, pygame.K_LEFT) and self.player.manual_vx < 0:
                        self.player.manual_vx = 0.0
                    elif event.key in (pygame.K_d, pygame.K_RIGHT) and self.player.manual_vx > 0:
                        self.player.manual_vx = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.toggle_rect.collidepoint(event.pos):
                    self.auto_mode = not self.auto_mode
                    self.player.manual_mode = not self.auto_mode
                    self.player.manual_vx = 0.0
                    self.player.manual_vy = 0.0
                    self.player._route = []
                    self.player._route_target = None
                    mode_str = "AUTO MODE" if self.auto_mode else "MANUAL PLAY"
                    print(f"  >> Control Mode: {mode_str}")
                elif self.rl_toggle_rect.collidepoint(event.pos):
                    self.rl_mode = not self.rl_mode
                    for g in self.ghosts.values():
                        g.rl_mode = self.rl_mode
                        if hasattr(g, "cbba_agent"):
                            g.cbba_agent.rl_mode = self.rl_mode
                    mode_str = "RL MODE ON" if self.rl_mode else "RL MODE OFF"
                    print(f"  >> Ghost Brain: {mode_str}")
                elif event.pos[0] > self.width and event.pos[1] < self.rows * self.cell_size:
                    # Clicking on right panel switches debug ghost view
                    self.debug_ghost_id = (self.debug_ghost_id + 1) % max(1, self.n_ghosts)
        return True

    def render_frame(self) -> bool:
        """Renders one visual frame in lockstep with MuJoCo simulation. Returns False on quit."""
        if not self.handle_events():
            return False

        # 1. Clear screen to solid black matching pacman.py:L1315
        self.screen.fill(BLACK)

        # 2. Draw continuous capsule maze & live pellets matching pacman.py:L1316
        self.draw_grid()

        # 3. Synchronize Ghost physical poses from MuJoCo runners and draw authentic ghosts
        for gid, runner in enumerate(self.engine.runners):
            ghost = self.ghosts[gid]
            gx_w, gy_w, _ = runner.position
            gx, gy = self.arena.to_grid(gx_w, gy_w)
            ghost.x = float(gx)
            ghost.y = float(gy)
            ghost.vx = math.cos(runner.yaw)
            ghost.vy = math.sin(runner.yaw)

        for ghost in self.ghosts.values():
            ghost.draw(self.screen)

        # 4. Synchronize Pac-Man physical pose from MuJoCo and draw mecanum rover car
        pac_wx, pac_wy, _ = self.engine.pacman_position
        px, py = self.arena.to_grid(pac_wx, pac_wy)
        self.player.x = float(px)
        self.player.y = float(py)
        self.player.powered = self.engine.is_pacman_powered
        self.player.power_timer = self.engine.power_timer_steps
        self.player.dead = (self.engine.outcome == "GHOSTS_WIN" and self.engine.game_over)
        self.player.score = self.engine.score
        self.player.draw(self.screen)

        # 5. Draw bottom HUD console matching pacman.py:L1320
        self.draw_hud()

        # 6. Draw personal map matching pacman.py:L1321
        self.draw_personal_map()

        # 7. Victory / Game Over overlay matching pacman.py:L1322-L1325
        if self.engine.game_over:
            if self.engine.outcome == "PACMAN_WIN":
                self.draw_overlay("CLEARED! Next map loading...", YELLOW)
            else:
                self.draw_overlay(f"GAME OVER   SCORE: {self.player.score}", RED)

        pygame.display.flip()
        return True

    def close(self):
        """Cleans up Pygame display resources."""
        try:
            pygame.quit()
        except Exception:
            pass

    def run(self, max_time: float = 300.0):
        """Runs the interactive simulation and visualizer loop standalone."""
        print(f"\n=================================================================")
        print(f"   PAC-MAN CONTINUOUS GAMESTATE VISUALIZER (MuJoCo Physics)     ")
        print(f"=================================================================")
        print(f"  Stage: {self.stage} ({self.cols}x{self.rows} grid, {self.n_ghosts} quadrupeds)")
        print(f"  Left Panel:  Continuous Map (pacman.py visualizer)")
        print(f"  Right Panel: Swarm cognitive radar & belief distribution")
        print(f"  Controls:    [WASD / Arrows] Drive Pac-Man  [M] Auto/Manual Mode")
        print(f"               [0-6] Switch ghost radar      [Space] Pause  [+/-] Speed  [Q] Quit")
        print(f"=================================================================\n")

        running = True
        try:
            while running and self.sim_time < max_time:
                # Step physics according to playback speed
                if not self.paused and not self.engine.game_over:
                    steps_to_take = max(1, int(self.speedup))
                    for _ in range(steps_to_take):
                        self.engine.step_simulation()
                        self.sim_time += self.step_dt
                        if self.engine.game_over:
                            break

                running = self.render_frame()
                if not running:
                    break
                self.clock.tick(50)  # 50 FPS screen refresh matching policy rate

        except KeyboardInterrupt:
            pass
        finally:
            pygame.quit()
            print(f"\nVisualizer closed at sim time: {self.sim_time:.2f}s | Score: {self.engine.score}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous Gamestate Visualizer for MuJoCo Pac-Man."
    )
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="Curriculum stage (1-4).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--speedup", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--max_time", type=float, default=300.0, help="Max run duration in simulated seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    vis = Pacman2DVisualizer(stage=args.stage, seed=args.seed, speedup=args.speedup)
    vis.run(max_time=args.max_time)


if __name__ == "__main__":
    main()
