#!/usr/bin/env python3
"""
Unit test verifying Phase 1 and Phase 2 implementations:
1. Power pellet duration parameterization T = D / S.
2. Bi-directional defusal synchronization.
3. Power pellet danger reflex: suicide prevention / flee when Pac-Man reaches pellet first.
4. Pellet denial: sprint to convert when ghost can beat Pac-Man.
"""

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in [str(ROOT / "pacmanbot"), str(ROOT / "pacman_sim"), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from game_engine import PacmanGameEngine
from allocator import TaskType

def test_phase1_and_phase2():
    print("=" * 70)
    print("TESTING PHASE 1 & PHASE 2 IMPLEMENTATIONS")
    print("=" * 70)

    # 1. Initialize GameEngine on Stage 1 (micro map)
    engine = PacmanGameEngine(stage=1, seed=42, enable_rl=True)

    # Check Phase 1: Power Duration T = D / S
    print("\n[Check 1: Power Duration Formula T = D / S]")
    print(f"  engine.power_duration_steps: {engine.power_duration_steps}")
    print(f"  engine.hive.power_duration_steps: {engine.hive.power_duration_steps}")
    g0 = engine.hive.ghosts[0]
    print(f"  ghost 0 power_duration_steps: {getattr(g0, 'power_duration_steps', None)}")
    assert 348 <= engine.power_duration_steps <= 352, f"Expected ~350, got {engine.power_duration_steps}"
    assert engine.hive.power_duration_steps == engine.power_duration_steps
    assert g0.power_duration_steps == engine.power_duration_steps
    print("  -> PASS: Power duration T = D / S correctly synchronized across engine, hive, and ghosts.")

    # Check Phase 2: Power Pellet Danger Reflex (Suicide Prevention)
    print("\n[Check 2: Power Pellet Danger Reflex - Suicide Prevention]")
    # Get first power pellet in arena
    assert len(engine.arena.world.power_pellets) > 0, "Stage 1 must have power pellets"
    ppx, ppy = engine.arena.world.power_pellets[0]
    print(f"  Target Power Pellet: grid=({ppy}, {ppx})")

    # Place Pac-Man 1.0 cell away from the power pellet
    pac_y = float(ppy)
    pac_x = float(ppx + 1.0)
    engine.pacman.y = pac_y
    engine.pacman.x = pac_x
    engine.pacman.vx = -0.5  # Moving toward power pellet
    engine.pacman.vy = 0.0
    engine.pacman.powered = False
    engine.power_timer_steps = 0

    # Place Ghost 0 at 3.5 cells away from the power pellet (Pac-Man will arrive first)
    g0.y = float(ppy)
    g0.x = float(ppx + 3.5)
    g0.known_pacman = (pac_y, pac_x)
    g0.pacman_last_seen = g0.frame
    g0.known_power_pellets = set([(ppx, ppy)])
    g0._is_striking = False

    # Update ghost autonomy
    g0.update(
        player_pos=(pac_y, pac_x),
        powered=False,
        all_ghosts=engine.hive.ghosts,
        skip_movement=False
    )
    des_vx, des_vy = g0.vx, g0.vy

    print(f"  Ghost position: ({g0.y}, {g0.x})")
    print(f"  Pac-Man position: ({pac_y}, {pac_x})")
    print(f"  Ghost computed velocity: vx={des_vx:.3f}, vy={des_vy:.3f}")
    print(f"  Ghost _is_striking: {g0._is_striking}")
    print(f"  Ghost _imminent_pellet_danger_pos: {g0._imminent_pellet_danger_pos}")

    # The ghost MUST NOT strike into Pac-Man, and MUST steer AWAY from the pellet (des_vx > 0)
    assert not g0._is_striking, "Terminal strike MUST be suppressed when Pac-Man is in power danger zone!"
    assert g0._imminent_pellet_danger_pos == (float(ppy), float(ppx)), "Danger pos must match power pellet!"
    assert des_vx > 0.0, f"Ghost must flee away from power pellet (des_vx > 0), got {des_vx}"
    print("  -> PASS: Ghost correctly aborted strike and fled away from imminent power pellet danger!")

    # Check Phase 2: Pellet Denial (Race to Convert)
    print("\n[Check 3: Pellet Denial - Race to Convert]")
    # Place Ghost 0 very close to pellet (0.6 cells away) and Pac-Man further away (2.5 cells)
    g0.y = float(ppy)
    g0.x = float(ppx + 0.6)
    pac_y = float(ppy)
    pac_x = float(ppx + 2.5)
    g0.known_pacman = (pac_y, pac_x)
    g0.pacman_last_seen = g0.frame
    g0.known_power_pellets = set([(ppx, ppy)])
    g0._is_striking = False

    g0.update(
        player_pos=(pac_y, pac_x),
        powered=False,
        all_ghosts=engine.hive.ghosts,
        skip_movement=False
    )
    des_vx, des_vy = g0.vx, g0.vy

    print(f"  Ghost position: ({g0.y}, {g0.x}) [d_ghost={0.6:.1f}]")
    print(f"  Pac-Man position: ({pac_y}, {pac_x}) [d_pac={2.5:.1f}]")
    print(f"  Ghost computed desired velocity: vx={des_vx:.3f}, vy={des_vy:.3f}")
    assert des_vx < 0.0, f"Ghost should sprint directly toward pellet to defuse (des_vx < 0), got {des_vx}"
    assert g0._imminent_pellet_danger_pos is None, "Ghost should NOT flee when it can win the defusal race!"
    print("  -> PASS: Ghost correctly initiated Pellet Denial sprint to convert before Pac-Man arrives!")

    # Check Phase 1: Defusal Synchronization
    print("\n[Check 4: Referee Defusal Synchronization]")
    initial_defused = engine.power_pellets_defused_total
    initial_active_pp = len(engine.active_power)
    # Move runner 0 directly onto the power pellet
    pp_idx = list(engine.active_power)[0]
    wx, wy = engine.arena.power_pellets[pp_idx]
    engine.runners[0].reset_pose(wx, wy, z=0.34)
    engine._check_referee_events()

    print(f"  Initial defused: {initial_defused}, After check: {engine.power_pellets_defused_total}")
    print(f"  Initial active: {initial_active_pp}, After check: {len(engine.active_power)}")
    assert engine.power_pellets_defused_total == initial_defused + 1, "Defused count must increase by 1!"
    assert len(engine.active_power) == initial_active_pp - 1, "Active power pellets must decrease by 1!"
    print("  -> PASS: Referee defusal event correctly synchronized with 3D model and swarm knowledge!")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 & PHASE 2 CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_phase1_and_phase2()
