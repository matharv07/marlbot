# xterra_mjlab

Locomotion reinforcement learning on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo-Warp). `xterra_mjlab` provides velocity-tracking training environments, a
robot registry, and train/play tooling built on mjlab and
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO.

This release includes the **SvanM2** quadruped on flat terrain.

> A **pretrained flat policy ships** under `checkpoints/svanm2_flat/`
> (`svanm2_flat.pt` for play, `policy.onnx` for deployment). You can `play` it
> immediately, or train your own. See `MODEL_CARD.md` for provenance and
> evaluation.

## Steps at a glance

1. Install mjlab (Prerequisites).
2. `pip install -e .` (Installation).
3. `list_envs.py` — confirm the task registers.
4. `play.py` — visualise the shipped policy (or train your own with `train.py`).

## Prerequisites

mjlab (and its MuJoCo-Warp / rsl_rl backend) runs on a CUDA GPU. This repository
is tested with:

| Component     | Version |
|---------------|---------|
| OS            | Ubuntu 22.04 |
| Python        | 3.11 |
| mjlab         | 1.5.2 |
| MuJoCo        | 3.10.0 |
| MuJoCo-Warp   | 3.10.0.2 |
| Warp          | 1.15.0 |
| rsl-rl-lib    | 5.4.0 |
| PyTorch       | 2.7.0 (CUDA 12.8) |
| Gymnasium     | 1.2.0 |

Install mjlab by following its documentation: https://github.com/mujocolab/mjlab

## Installation

Activate the Python environment that has mjlab, then install this package in
editable mode:

```bash
git clone https://github.com/xterra-robotics/xterra_mjlab.git
cd xterra_mjlab
pip install -e .
```

Verify the task is registered:

```bash
python scripts/list_envs.py
```

Expected output:

```
Registered xTerra mjlab tasks:
  xTerra-Mjlab-Velocity-Flat-SvanM2
```

## Usage

### Train

```bash
python scripts/train.py xTerra-Mjlab-Velocity-Flat-SvanM2 --env.scene.num-envs 4096
```

Checkpoints, TensorBoard logs, and exported `policy.onnx` files are written under
`logs/`. A quick pipeline smoke test uses fewer envs and iterations:

```bash
python scripts/train.py xTerra-Mjlab-Velocity-Flat-SvanM2 \
    --env.scene.num-envs 64 --agent.max-iterations 10
```

### Play

Visualise the **shipped** policy:

```bash
python scripts/play.py xTerra-Mjlab-Velocity-Flat-SvanM2 \
    --checkpoint-file checkpoints/svanm2_flat/svanm2_flat.pt
```

Or point `--checkpoint-file` at a `model_*.pt` from a run you trained.

Test the environment with no policy:

```bash
python scripts/play.py xTerra-Mjlab-Velocity-Flat-SvanM2 --agent zero
```

### Multi-Agent Pac-Man Simulation (Continuous MuJoCo + pacman.py)

Launch the multi-agent Pac-Man simulation coupling SvanM2 quadruped locomotion in MuJoCo with the synchronized 2D continuous `pacman.py` visualizer:

```bash
# Launch both MuJoCo 3D viewer & continuous map pacman.py window simultaneously
python launch.py
python launch.py --stage 1
python launch.py --stage 2 --speedup 2.0

# 2D continuous visualizer only
python launch.py --gui 2d

# MuJoCo 3D viewer only
python launch.py --gui 3d

# Headless benchmark
python launch.py --headless --max_time 30
```

## Tasks

| Task ID | Description |
|---------|-------------|
| `xTerra-Mjlab-Velocity-Flat-SvanM2` | Flat-terrain velocity tracking, MLP actor-critic PPO. |

## Repository layout

```
xterra_mjlab/
├── xterra_mjlab/
│   ├── actuators/svanm2_actuator.py   # parallel-mechanism actuator
│   ├── assets/svanm2/                 # robot constants + MJCF + STL meshes
│   └── tasks/velocity/config/svanm2/  # env cfg, RL cfg, task registration
├── scripts/                           # train / play / list_envs
├── checkpoints/                       # published policies
└── licenses/                          # third-party license inventory
```

## Adding a robot

Add a package under `xterra_mjlab/assets/` (robot constants + actuator + MJCF)
and a package under `xterra_mjlab/tasks/velocity/config/`, mirroring `svanm2`,
then import it from `xterra_mjlab/tasks/velocity/config/__init__.py`.

## The SvanM2 actuator

The SvanM2 leg is a parallel mechanism: a belt routes from the thigh actuator
past the knee to drive the calf, so the calf-motor angle depends on both the
calf and thigh joint angles. `SvanM2Actuator`
(`xterra_mjlab/actuators/svanm2_actuator.py`) models this transmission — joint
positions/velocities and torques are mapped through the per-leg Jacobian,
motor-space PD gains are derived from the joint-space configuration gains and
gear ratios, and an action-delay model sits on top. All 12 leg joints live in a
single actuator group so the thigh↔calf belt coupling is captured. See the
module docstring for the full derivation.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Third-party components (mjlab, rsl_rl,
MuJoCo/Warp) are installed separately; their licenses are collected under
`licenses/`.
