"""
simulate.py — Main Entrypoint for Native MuJoCo Continuous Pac-Man Simulation.

Features:
  • Runs 100% native MuJoCo multi-agent simulation (zero Gazebo / zero ROS 2)
  • SvanM2 quadruped robots trotting with xterra_mjlab RL policy at 50 Hz
  • Decentralized multi-agent ghost swarm with local LiDAR fog-of-war, 12m NRF24 mesh & CBBA
  • Interactive 3D passive viewer with real-time HUD or high-speed headless execution
  • Procedural continuous arenas generated per run

Usage:
  python pacman_sim/simulate.py --stage 1
  python pacman_sim/simulate.py --stage 2 --seed 42
  python pacman_sim/simulate.py --stage 3 --headless --max_time 60
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add repos to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMANBOT_DIR = _REPO_ROOT / "pacmanbot"
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"

for p in [str(_PACMANBOT_DIR), str(_PACMAN_SIM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco
import mujoco.viewer
from game_engine import PacmanGameEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native MuJoCo Continuous Pac-Man Multi-Agent Simulation with SvanM2 Quadrupeds."
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Curriculum stage (1: 17x13/2 ghosts, 2: 27x21/5 ghosts, 3: 33x27/6 ghosts, 4: 41x33/7 ghosts).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for procedural maze generation (random per run if omitted).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Metric scaling factor (meters per grid unit, default: 1.5).",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=180.0,
        help="Maximum simulation run time in seconds (default: 180s).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="checkpoints/svanm2_flat/svanm2_flat.pt",
        help="Path to SvanM2 flat locomotion policy checkpoint.",
    )
    parser.add_argument(
        "--gui",
        type=str,
        default="both",
        choices=["both", "3d", "2d"],
        help="Visualizer interface: 'both' (launches both native MuJoCo 3D viewer & continuous map pacman.py Pygame display, matching minibot), '3d' (native MuJoCo 3D viewer only), or '2d' (Pygame dual-panel visualizer only).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless without opening interactive viewer.",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Simulation playback speedup multiplier (1.0 = real-time, 2.0 = 2x speed, 0 = uncapped max speed).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=300.0,
        help="Target visual render framerate for display (default: 300.0 for 300 Hz monitor, 60.0 for standard displays).",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint number to load from pacmanbot/checkpoints/ (e.g. 1000 for ckpt_1000.pt, -1 for latest).",
    )
    parser.add_argument(
        "--no_rl",
        action="store_true",
        help="Disable MAPPO GhostActor neural network and run in purely heuristic CBBA mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Align SDL2 video driver on Wayland to prevent EGL context collisions with GLFW
    if os.environ.get("XDG_SESSION_TYPE") == "wayland" and "WAYLAND_DISPLAY" in os.environ:
        os.environ.setdefault("SDL_VIDEODRIVER", "wayland")

    policy_path = _REPO_ROOT / args.policy
    assert policy_path.exists(), f"Policy checkpoint not found at {policy_path}"

    print("=================================================================")
    print("   NATIVE MUJOCO CONTINUOUS PAC-MAN MULTI-AGENT SIMULATION       ")
    print("=================================================================")
    print(f"  Stage: {args.stage}")
    print(f"  Seed: {'Random' if args.seed is None else args.seed}")
    print(f"  Scale Factor: {args.scale:.2f} m/unit")
    print(f"  Max Run Time: {args.max_time:.1f} s")
    print(f"  Playback Speed: {'Uncapped Max' if args.speedup <= 0 else f'{args.speedup:.1f}x'}")
    fps_desc = f"{args.fps:.0f} FPS (High-Refresh 300 Hz Mode)" if args.fps >= 240 else f"{args.fps:.0f} FPS"
    print(f"  Target Framerate: {fps_desc}")
    print(f"  Policy: {policy_path}")
    mode_str = (
        "Headless"
        if args.headless
        else (
            "Interactive 2D Visualizer"
            if args.gui == "2d"
            else (
                "Dual Synchronized Viewers (MuJoCo 3D + Continuous Map pacman.py)"
                if args.gui == "both"
                else "Interactive 3D Viewer"
            )
        )
    )
    print(f"  Mode: {mode_str}")
    print("=================================================================")

    if not args.headless and args.gui == "2d":
        from visualizer import Pacman2DVisualizer
        vis = Pacman2DVisualizer(stage=args.stage, seed=args.seed, speedup=args.speedup)
        vis.run(max_time=args.max_time)
        return

    # Initialize complete game engine
    print("[1/2] Initializing continuous procedural arena and agents...")
    t0_init = time.time()
    engine = PacmanGameEngine(
        stage=args.stage,
        seed=args.seed,
        scale=args.scale,
        policy_ckpt=str(policy_path),
        ghost_ckpt=args.checkpoint,
        enable_rl=not args.no_rl,
    )
    t_init = time.time() - t0_init
    print(f"  Initialized in {t_init:.2f}s!")
    print(f"  Arena size: {engine.arena.width_m:.1f}m x {engine.arena.height_m:.1f}m")
    print(f"  Ghosts spawned: {engine.hive.alive_ghost_count} SvanM2 quadrupeds")
    print(f"  Regular pellets: {len(engine.active_pellets)}")
    print(f"  Power pellets: {len(engine.active_power)}")
    print(f"  Pac-Man start: ({engine.arena.pacman_start[0]:.2f}, {engine.arena.pacman_start[1]:.2f})")
    print("=================================================================")

    step_dt = 0.020  # 50 Hz policy update step
    sim_time = 0.0
    start_wall_time = time.time()

    if args.headless:
        print("[2/2] Running headless simulation...")
        steps_total = int(args.max_time / step_dt)
        last_log = time.time()

        for step in range(steps_total):
            engine.step_simulation()
            sim_time += step_dt

            # Progress log every 5 simulated seconds
            if step % 250 == 0:
                now = time.time()
                fps = 250.0 / max(1e-4, (now - last_log))
                last_log = now
                pac_pos = engine.pacman_position
                pow_str = "POWERED" if engine.is_pacman_powered else "NORMAL"
                print(
                    f"[{sim_time:5.1f}s | {fps:5.1f} steps/s] Score: {engine.score:4d} | "
                    f"Pac-Man: ({pac_pos[0]:5.1f}, {pac_pos[1]:5.1f}) [{pow_str:7s}] | "
                    f"Pellets: {len(engine.active_pellets):3d} | "
                    f"Power: {len(engine.active_power):2d} | "
                    f"Ghosts alive: {engine.hive.alive_ghost_count}/{engine.arena.n_ghosts}"
                )

            if engine.game_over:
                print(f"\nGame Over at {sim_time:.2f}s! Outcome: {engine.outcome}")
                break

    else:
        print("\n  [Controls]")
        print("    • Mouse: Left-drag to orbit camera, Right-drag to pan, Scroll to zoom.")
        print("    • Keyboard: [+] Speed up  [-] Slow down  [Space] Pause/Resume  [Tab] Toggle sidebars.")
        print("    • Full Screen: Press [Super + Up Arrow] or click window Maximize button to fill the screen.")
        print("  [Visuals]")
        print("    • SvanM2 quadrupeds: Authentic in-game ghost body colors (Red, Pink, Cyan, Orange, Purple, Green, Gold).")
        print("    • Pac-Man: Mobile robot with 3D LiDAR puck & visor (surges electric blue when powered)!\n")

        # Interactive state
        speed_state = {"speedup": args.speedup, "paused": False}
        vis = None

        def on_key(keycode: int):
            # GLFW keycodes: 61 is '=', 43 is '+', 45 is '-', 32 is Space
            if keycode in (43, 61):  # '+' or '='
                if speed_state["speedup"] <= 0:
                    speed_state["speedup"] = 2.0
                else:
                    speed_state["speedup"] = min(8.0, speed_state["speedup"] + 0.5)
                print(f"  >> Playback speed: {speed_state['speedup']:.1f}x")
                if vis is not None:
                    vis.speedup = speed_state["speedup"]
            elif keycode in (45, 95):  # '-' or '_'
                speed_state["speedup"] = max(0.5, speed_state["speedup"] - 0.5)
                print(f"  >> Playback speed: {speed_state['speedup']:.1f}x")
                if vis is not None:
                    vis.speedup = speed_state["speedup"]
            elif keycode == 32:  # Space
                speed_state["paused"] = not speed_state["paused"]
                state_str = "PAUSED" if speed_state["paused"] else "RESUMED"
                print(f"  >> Simulation {state_str}")
                if vis is not None:
                    vis.paused = speed_state["paused"]

        with mujoco.viewer.launch_passive(
            engine.model,
            engine.data,
            key_callback=on_key,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            # Initialize 2D continuous map pacman visualizer AFTER GLFW context is established
            if not args.headless and args.gui == "both":
                from visualizer import Pacman2DVisualizer
                vis = Pacman2DVisualizer(
                    stage=args.stage,
                    seed=args.seed,
                    speedup=args.speedup,
                    engine=engine,
                    speed_state=speed_state,
                )

            # Cinematic angled view above arena
            max_dim = max(engine.arena.width_m, engine.arena.height_m)
            viewer.cam.distance = max_dim * 0.82
            viewer.cam.elevation = -48.0
            viewer.cam.azimuth = 90.0
            viewer.cam.lookat = [0.0, 0.0, 0.0]

            ghost_colors = ["Red", "Pink", "Cyan", "Orange", "Purple", "Green", "Gold"]
            last_status_print = 0.0

            # Target visual rendering rate (supports up to 300 Hz displays) and decoupled simulation accumulator
            target_fps = max(30.0, float(args.fps))
            frame_dt = 1.0 / target_fps
            sim_accumulator = 0.0
            last_wall_time = time.time()
            rtf_window_start = time.time()
            rtf_sim_start = sim_time
            current_rtf = speed_state["speedup"]

            while viewer.is_running() and sim_time < args.max_time:
                loop_start = time.time()
                wall_dt = loop_start - last_wall_time
                last_wall_time = loop_start

                # Prevent spiral-of-death during window dragging or OS hiccups
                wall_dt = min(wall_dt, 0.1)
                cur_spd = speed_state["speedup"]

                if not speed_state["paused"]:
                    if cur_spd <= 0:
                        # Uncapped MAX speed: execute batch of physics steps per visual frame
                        for _ in range(8):
                            if sim_time >= args.max_time or engine.game_over:
                                break
                            engine.step_simulation()
                            sim_time += step_dt
                    else:
                        # Decoupled accumulator: advance as many 20ms steps as needed to meet target speed
                        sim_accumulator += wall_dt * cur_spd
                        steps_this_frame = 0
                        # Cap at 10 steps per visual frame to preserve viewer responsiveness
                        while sim_accumulator >= step_dt and steps_this_frame < 10:
                            if sim_time >= args.max_time or engine.game_over:
                                break
                            engine.step_simulation()
                            sim_time += step_dt
                            sim_accumulator -= step_dt
                            steps_this_frame += 1

                        # Clamp accumulator if hardware cannot keep up with high multiplier
                        if sim_accumulator > step_dt * 2:
                            sim_accumulator = step_dt

                # Sync visual state to MuJoCo viewer once per frame
                viewer.sync()

                # Sync visual state to 2D continuous pacman visualizer
                if vis is not None:
                    vis.paused = speed_state["paused"]
                    vis.speedup = speed_state["speedup"]
                    vis.sim_time = sim_time
                    if not vis.render_frame():
                        break

                # Calculate realized RTF periodically (every 1.0s)
                now = time.time()
                if now - rtf_window_start >= 1.0:
                    dt_win = now - rtf_window_start
                    d_sim = sim_time - rtf_sim_start
                    current_rtf = d_sim / dt_win if dt_win > 0 else 0.0
                    rtf_window_start = now
                    rtf_sim_start = sim_time

                # Print periodic HUD status to console
                if now - last_status_print >= 2.0:
                    last_status_print = now
                    pac_pos = engine.pacman_position
                    pow_str = f"POWERED ({engine.power_timer_steps * step_dt:.1f}s)" if engine.is_pacman_powered else "NORMAL"
                    
                    ghost_status = []
                    for gid, g in engine.hive.ghosts.items():
                        col_name = ghost_colors[gid % len(ghost_colors)]
                        state_str = "Dead" if g.dead else "Alive"
                        ghost_status.append(f"G{gid}({col_name}): {state_str}")
                    
                    ghost_hud = " | ".join(ghost_status)
                    spd_tgt_str = f"{cur_spd:.1f}x" if cur_spd > 0 else "MAX"
                    print(
                        f"[{sim_time:5.1f}s | RTF: {current_rtf:4.2f}x (Target: {spd_tgt_str})] Score: {engine.score:4d} | Pac-Man: [{pow_str:13s}] | "
                        f"Pellets: {len(engine.active_pellets):3d} | Power: {len(engine.active_power):2d} | Alive: {engine.hive.alive_ghost_count}/{engine.arena.n_ghosts}\n"
                        f"       Ghosts: {ghost_hud}"
                    )

                if engine.game_over:
                    print(f"\nGame Ended at {sim_time:.2f}s! Outcome: {engine.outcome}")
                    time.sleep(2.0)
                    break

                # Frame pacing: sleep only the remainder of visual frame (supports 300 Hz with 0.4ms sleep threshold)
                if cur_spd > 0:
                    elapsed_frame = time.time() - loop_start
                    sleep_rem = frame_dt - elapsed_frame
                    if sleep_rem > 0.0004:
                        time.sleep(sleep_rem)

        if vis is not None:
            vis.close()

    total_wall = time.time() - start_wall_time
    print("=================================================================")
    print("                      SIMULATION REPORT                          ")
    print("=================================================================")
    print(f"  Outcome:                {engine.outcome}")
    print(f"  Final Score:            {engine.score}")
    print(f"  Simulated Duration:     {sim_time:.2f} s")
    print(f"  Wall-Clock Duration:    {total_wall:.2f} s")
    print(f"  Regular Pellets Left:   {len(engine.active_pellets)}")
    print(f"  Power Pellets Left:     {len(engine.active_power)}")
    print(f"  Ghosts Alive:           {engine.hive.alive_ghost_count}/{engine.arena.n_ghosts}")
    print("=================================================================")


if __name__ == "__main__":
    main()
