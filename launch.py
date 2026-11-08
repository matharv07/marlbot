#!/usr/bin/env python3
"""
launch.py — Unified Launch Entrypoint for MuJoCo Continuous Pac-Man Simulation.

Launches both MuJoCo 3D Physics Simulation (mjlab) and the 2D Continuous Map pacman.py
Gamestate Visualizer synchronized in real-time, replicating the minibot architecture.

Usage:
  python launch.py
  python launch.py --stage 1
  python launch.py --stage 2 --speedup 2.0
  python launch.py --stage 4 --seed 42
  python launch.py --gui 3d       # MuJoCo 3D only
  python launch.py --gui 2d       # 2D visualizer only
  python launch.py --headless     # Headless benchmark
"""

import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"

for p in [str(_PACMANBOT_DIR), str(_PACMAN_SIM_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from simulate import main

if __name__ == "__main__":
    main()
