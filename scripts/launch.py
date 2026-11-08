#!/usr/bin/env python3
"""
scripts/launch.py — Unified Launch Entrypoint for MuJoCo Continuous Pac-Man Simulation.
"""

import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"

for p in [str(_PACMANBOT_DIR), str(_PACMAN_SIM_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from simulate import main

if __name__ == "__main__":
    main()
