#!/usr/bin/env python3
"""
compare_gameplay.py — Head-to-head comparison between pacmanbot/pacman.py and pacman_sim.

Loads the exact same field, seeds, and agents in both engines, runs 20 frames,
and logs side-by-side moves, tasks, headings, and speeds.
"""

import sys
import os
import random
from pathlib import Path

# Add directories to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"

for p in [str(_PACMANBOT_DIR), str(_PACMAN_SIM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(str(_PACMANBOT_DIR))

import numpy as np
import torch
import pacman
from game_engine import PacmanGameEngine


def run_comparison(seed: int = 123, n_frames: int = 20):
    print("=" * 75)
    print("   HEAD-TO-HEAD CONTROLLER COMPARISON: pacman.py vs pacman_sim   ")
    print("=" * 75)
    print(f"Seed: {seed} | Frames: {n_frames}")

    # 1. Initialize pacmanbot.pacman.Game
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pacman.RL_MODE = True
    game_bot = pacman.Game()

    print(f"\n[pacman.py] Initialized:")
    print(f"  Grid: {len(game_bot.grid)} rows x {len(game_bot.grid[0])} cols")
    print(f"  Pellets: {len(game_bot.world.pellets)} regular, {len(game_bot.world.power_pellets)} power")
    print(f"  Player start (r, c): {game_bot.player_start}")
    print(f"  Ghosts: {len(game_bot.ghosts)}")

    # 2. Extract field and starts to pass into PacmanGameEngine
    ghost_starts_grid = [(float(g.y), float(g.x)) for g in game_bot.ghosts.values()]

    # 3. Initialize pacman_sim.game_engine.PacmanGameEngine with the EXACT SAME FIELD
    engine = PacmanGameEngine(
        stage=4,
        scale=1.5,
        policy_ckpt=str(_REPO_ROOT / "checkpoints/svanm2_flat/svanm2_flat.pt"),
        ghost_ckpt=-1,
        enable_rl=True,
        world=game_bot.world,
        grid=game_bot.grid,
        player_start_grid=game_bot.player_start,
        ghost_starts_grid=ghost_starts_grid,
    )

    print(f"\n[pacman_sim] Initialized with identical field and starts.")

    print("\n" + "=" * 75)
    print(f"{'F':>2} | {'Engine':<10} | {'G0 Pos (x, y)':<18} | {'G0 Vel (vx, vy)':<18} | {'Task':<20} | {'Speed'}")
    print("-" * 75)

    for f in range(n_frames):
        # Synchronize torch seed before decision frame (frame 5, 11, 17, ...)
        if (f + 1) % 6 == 0:
            torch.manual_seed(seed * 1000 + f)

        # Step pacmanbot
        game_bot.update()
        bot_g0 = game_bot.ghosts[0]
        bot_task = bot_g0.cbba_agent.get_active_task()
        bot_task_str = f"{bot_task.task_type.name if hasattr(bot_task.task_type, 'name') else bot_task.task_type} {bot_task.target_pos}" if bot_task else "None"
        bot_spd = getattr(bot_g0, 'current_speed_mult', 1.0)

        if (f + 1) % 6 == 0:
            torch.manual_seed(seed * 1000 + f)

        # Step pacman_sim
        engine.step_simulation()
        sim_g0 = engine.hive.ghosts[0]
        sim_task = sim_g0.cbba_agent.get_active_task()
        sim_task_str = f"{sim_task.task_type.name if hasattr(sim_task.task_type, 'name') else sim_task.task_type} {sim_task.target_pos}" if sim_task else "None"
        sim_spd = getattr(sim_g0, 'current_speed_mult', 1.0)

        # Print comparison for Ghost 0
        print(f"{f:2d} | {'pacman.py':<10} | ({bot_g0.x:6.2f}, {bot_g0.y:6.2f})   | ({bot_g0.vx:6.2f}, {bot_g0.vy:6.2f})   | {bot_task_str:<20} | {bot_spd:.2f}x")
        print(f"   | {'pacman_sim':<10} | ({sim_g0.x:6.2f}, {sim_g0.y:6.2f})   | ({sim_g0.vx:6.2f}, {sim_g0.vy:6.2f})   | {sim_task_str:<20} | {sim_spd:.2f}x")
        print("-" * 75)

    print("\nComparison complete!")


if __name__ == "__main__":
    run_comparison(seed=123, n_frames=18)
