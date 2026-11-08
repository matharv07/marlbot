# Native MuJoCo Continuous Multi-Agent Pac-Man Simulation

High-performance continuous multi-agent Pac-Man simulation running 100% natively in **MuJoCo 3.x** / **mjlab**. Fully replaces the deprecated Gazebo stack with native GPU/CPU multi-agent physics and zero ROS 2 overhead.

---

## Key Features

1. **Native MuJoCo Physics & Locomotion**:
   - Multiple **SvanM2** quadrupeds simulated in a shared continuous arena.
   - Evaluates the pre-trained `svanm2_flat.pt` locomotion policy at **50 Hz** using the exact 270-dim term-major observation layout.
   - Computes parallel-mechanism belt transmission torques at **500 Hz** ($K_p = 20.0, K_d = 0.7$, belt coupling $0.5$).
   - SvanM2 bodies sport authentic in-game ghost colors (Red, Pink, Cyan, Orange, Purple, Green, Gold) with dark metallic mechanical limbs. Ghosts **never turn blue**; they maintain their signature colors at all times.
   - Omnidirectional coordinate frame mapping (`cmd.linear.x = vx * cos_y + vy * sin_y`, `cmd.linear.y = -vx * sin_y + vy * cos_y`) prevents corner stalling, maintaining 100% velocity magnitude fluidly through turns.

2. **Articulated Autonomous Pac-Man Robot**:
   - High-fidelity autonomous mobile robot chassis with aerodynamic dome, dark protective bumper ring, left/right drive wheels with gold hubs, and smooth ground casters.
   - Equipped with a roof-mounted 3D LiDAR puck (glowing optical window) and front stereo camera visor with twin optical eyes and LED headlights.
   - Continuous heading orientation smoothly steers the robot and face sensors along the travel vector.
   - Surges electric blue during power pellet frenzy (`POWERED_COLOR = [0.0, 0.47, 1.0]`).

3. **Multi-Agent Autonomy & Fog of War (Parity with `pacmanbot` & `minibot`)**:
   - **Local Perception Only**: Ghosts have **NO global knowledge** of Pac-Man's power state. A ghost only discovers Pac-Man is powered if Pac-Man is inside its direct **90-ray LiDAR line-of-sight**.
   - **Simulated 12m NRF24 Radio Mesh**: Sighting packets `("pacman", r, c, powered, obs_frame)` are broadcast only to peers within communication radius with 2-hop mesh relaying.
   - **CBBA Decentralized Task Allocation**: Swarm bids on `HUNT`, `FLANK`, `CONVERT`, and `EVADE_TRACK` tasks using Consensus-Based Bundle Auctions.
   - **Power Pellet Defusal (CONVERT Rule)**: When a ghost reaches an uncollected power pellet (1.20m threshold matching 0.8 units in 1.5m scale), it defuses it into a standard 10-point pellet, updates the world model arrays, and broadcasts the conversion to the swarm.
   - **16-Ray Context Steering & Calibrated Speeds**: Continuous obstacle avoidance, peer repulsion, and smooth steering. Speeds are calibrated to the exact 1.3x to 2.0x curriculum ratio relative to SvanM2 sprint speeds.

4. **Procedural Mazes**:
   - Generates unique continuous mazes per run (capsule walls clipped to boundary via Liang-Barsky algorithm).
   - Random procedural generation by default or deterministic with `--seed <int>`.

---

## Quick Start

Navigate to the repository root:
```bash
cd ~/xTerra/xterra_mjlab
```

### 1. Interactive 3D Simulation (with MuJoCo Passive Viewer)
Run the active conda/venv environment (`xterra`):

```bash
/home/atharv/Downloads/ENTER/envs/xterra/bin/python pacman_sim/simulate.py --stage 1
```

Run with 5 quadrupeds (Stage 2) or 7 quadrupeds (Stage 4) with a specific seed:
```bash
/home/atharv/Downloads/ENTER/envs/xterra/bin/python pacman_sim/simulate.py --stage 2 --seed 42
/home/atharv/Downloads/ENTER/envs/xterra/bin/python pacman_sim/simulate.py --stage 4 --seed 123
```

*(Alternatively, you can run from anywhere using the absolute path to `simulate.py`):*
```bash
/home/atharv/Downloads/ENTER/envs/xterra/bin/python ~/xTerra/xterra_mjlab/pacman_sim/simulate.py --stage 4 --seed 123
```

### 1. Dual Synchronized Viewers (MuJoCo 3D + Continuous Map pacman.py)
Launch both the native MuJoCo 3D viewer and the continuous 2D `pacman.py` Pygame window simultaneously in lockstep (replicating `minibot`):

```bash
python launch.py
# Or with specific stage and seed:
python launch.py --stage 1
python launch.py --stage 2 --seed 42
python launch.py --stage 4 --speedup 2.0
```

### 2. Interactive 2D Gamestate Visualizer Standalone
Run the dual-panel 2D continuous visualizer directly based on `pacmanbot/pacman.py`:

```bash
python pacman_sim/visualizer.py --stage 1
# Or via simulate.py / launch.py:
python launch.py --stage 2 --gui 2d
```

### 3. Native MuJoCo 3D Viewer Only
```bash
python launch.py --stage 1 --gui 3d
```

### 4. High-Speed Headless Benchmark
Run headless simulation (runs at **>180-250 FPS**, up to 4x faster than real-time):

```bash
python launch.py --stage 2 --headless --max_time 30
```

---

## CLI Options & Controls

| Argument | Default | Description |
|---|---|---|
| `--stage` | `1` | Curriculum stage: `1` (2 ghosts), `2` (5 ghosts), `3` (6 ghosts), `4` (7 ghosts) |
| `--gui` | `both` | Interface mode: `both` (dual MuJoCo 3D + continuous pacman.py), `3d` (3D only), or `2d` (Pygame only) |
| `--speedup` | `1.0` | Playback speedup multiplier: `1.0` (real-time), `2.0` (2x speed), `0` (uncapped max FPS) |
| `--seed` | `None` | Procedural maze seed (`None` generates a fresh maze each run) |
| `--scale` | `1.5` | Meters per continuous grid unit |
| `--max_time` | `180.0` | Maximum simulation duration in seconds |
| `--policy` | `checkpoints/svanm2_flat/svanm2_flat.pt` | Path to trained policy checkpoint |
| `--headless` | `False` | Run headless without opening interactive viewer |

### 2D Visualizer Controls (`visualizer.py` or `--gui 2d`)
- **[0-6] Number Keys**: Switch cognitive radar view to inspect any ghost's personal map and belief distribution.
- **[Space]**: Pause / Resume simulation.
- **[+] / [-]**: Speed up / slow down playback speed.
- **[R]**: Reset and generate a fresh maze.
- **[Q] / [Esc]**: Exit cleanly.

### 3D Viewer Controls (`simulate.py --gui 3d`)
- **Full Screen**: Press `Super + Up Arrow` (or click window Maximize / double-click title bar) on Ubuntu/Gnome.
- **Speed Up / Slow Down**: Press `+` or `=` to increase playback speed (up to 8x), and `-` or `_` to slow down.
- **Pause / Resume**: Press `Space`.
- **Toggle Sidebars**: Press `Tab` or `F1` to toggle MuJoCo debug sidebars on/off.
- **Camera**: Left-click + drag to orbit, Right-click + drag to pan, Scroll to zoom.

## Architecture

- [`arena_builder.py`](file:///home/atharv/xTerra/xterra_mjlab/pacman_sim/arena_builder.py): Generates procedural continuous mazes and compiles `MjModel` via `MjSpec.attach`.
- [`svanm2_runner.py`](file:///home/atharv/xTerra/xterra_mjlab/pacman_sim/svanm2_runner.py): Controls each quadruped's 50 Hz policy and 500 Hz belt transmission PD loop.
- [`hive_coordinator.py`](file:///home/atharv/xTerra/xterra_mjlab/pacman_sim/hive_coordinator.py): Decentralized multi-agent swarm manager (`pacmanbot.ghost.Ghost` with LiDAR, NRF24 radio mesh, and CBBA auctions).
- [`game_engine.py`](file:///home/atharv/xTerra/xterra_mjlab/pacman_sim/game_engine.py): Pac-Man autonomous navigation, doorway traversal, pellet collection, power defusal, and contact referee.
- [`visualizer.py`](file:///home/atharv/xTerra/xterra_mjlab/pacman_sim/visualizer.py): High-performance dual-panel 2D visualizer with global arena view and ghost swarm cognitive radar.
- [`simulate.py`](file:///home/atharv/xTerra/xterra_mjlab/pacman_sim/simulate.py): Main entrypoint supporting 3D viewer, 2D visualizer, and headless modes.
