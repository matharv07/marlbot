#!/usr/bin/env python3
"""
svanm2_runner.py — High-fidelity SvanM2 Quadruped Controller for MuJoCo.

Executes the trained RL locomotion policy (svanm2_flat.pt) at 50 Hz and
computes parallel-mechanism belt transmission torques at 500 Hz:
  • Exact 270-dim term-major observation history (6 frames x 45 dims)
  • Belt-coupled drivetrain Jacobian J_theta and joint-to-motor PD law
  • Omnidirectional base-frame velocity tracking with yaw heading regulation
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import mujoco

# Default stance joint angles (radians) for SvanM2
DEFAULT_QPOS = np.array([
    0.0,  0.5806, -1.1716,  # FL: hip, thigh, calf
    0.0,  0.5806, -1.1716,  # FR: hip, thigh, calf
    0.0,  0.7167, -1.1342,  # RL: hip, thigh, calf
    0.0,  0.7167, -1.1342   # RR: hip, thigh, calf
], dtype=np.float32)

# Action scaling factors from mjlab convention (0.25 * effort / kp)
ACTION_SCALE = np.array([
    0.15, 0.15, 0.30,  # FL: hip, thigh, calf
    0.15, 0.15, 0.30,  # FR: hip, thigh, calf
    0.15, 0.15, 0.30,  # RL: hip, thigh, calf
    0.15, 0.15, 0.30   # RR: hip, thigh, calf
], dtype=np.float32)

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

ACTUATOR_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]


class SvanM2Policy(nn.Module):
    """Loads and evaluates the trained locomotion policy MLP."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(270, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12)
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = {k: v for k, v in ckpt["actor_state_dict"].items() if not k.startswith("distribution")}
        self.load_state_dict(sd)
        self.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class SvanM2Runner:
    """Controls a single SvanM2 robot instance within an MjModel/MjData composite simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gid: int,
        prefix: str,
        policy: SvanM2Policy,
    ):
        self.model = model
        self.data = data
        self.gid = gid
        self.prefix = prefix
        self.policy = policy

        # Resolve robot base joint (freejoint)
        root_joint_name = f"{prefix}root" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}root") != -1 else prefix.rstrip("/")
        self.root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        if self.root_jnt_id == -1:
            # Fallback to search any free joint with prefix
            for j in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if name and name.startswith(prefix) and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    self.root_jnt_id = j
                    break

        assert self.root_jnt_id != -1, f"Could not find root joint for {prefix}"
        self.root_qpos_adr = model.jnt_qposadr[self.root_jnt_id]
        self.root_dof_adr = model.jnt_dofadr[self.root_jnt_id]

        # Resolve base body
        base_body_name = f"{prefix}base"
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)

        # Resolve joint indices
        self.joint_qpos_adrs = []
        self.joint_dof_adrs = []
        for jname in JOINT_NAMES:
            full_name = f"{prefix}{jname}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
            assert jid != -1, f"Joint {full_name} not found in model"
            self.joint_qpos_adrs.append(model.jnt_qposadr[jid])
            self.joint_dof_adrs.append(model.jnt_dofadr[jid])
        self.joint_qpos_adrs = np.array(self.joint_qpos_adrs, dtype=np.int32)
        self.joint_dof_adrs = np.array(self.joint_dof_adrs, dtype=np.int32)

        # Resolve actuator indices
        self.actuator_adrs = []
        for aname in ACTUATOR_NAMES:
            full_name = f"{prefix}{aname}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full_name)
            assert aid != -1, f"Actuator {full_name} not found in model"
            self.actuator_adrs.append(aid)
        self.actuator_adrs = np.array(self.actuator_adrs, dtype=np.int32)

        # Resolve IMU angular velocity sensor
        imu_name = f"{prefix}imu_ang_vel"
        imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, imu_name)
        assert imu_id != -1, f"Sensor {imu_name} not found in model"
        self.imu_sensor_adr = model.sensor_adr[imu_id]

        # Build drivetrain transmission matrices
        self._build_transmission()

        # Observation history buffers (6 frames each, chronological order)
        self.hist_len = 6
        self.reset_history()

        # Velocity command [vx, vy, wz] in robot base frame
        self.cmd_base = np.zeros(3, dtype=np.float32)
        self.target_cmd = np.zeros(3, dtype=np.float32)
        self.filtered_target_yaw: Optional[float] = None

        # PD position target
        self.q_target = DEFAULT_QPOS.copy()
        self.last_action = np.zeros(12, dtype=np.float32)

    def _build_transmission(self):
        """Builds parallel mechanism belt Jacobian and motor stiffness gains."""
        self.J = np.zeros((12, 12), dtype=np.float32)
        gear = [8.0, 8.0, 16.0]
        hk = 0.5
        motor_gear = np.zeros(12, dtype=np.float32)
        self.force_limit = np.zeros(12, dtype=np.float32)

        for leg in range(4):
            base = leg * 3
            self.J[base + 0, base + 0] = gear[0]
            self.J[base + 1, base + 1] = gear[1]
            self.J[base + 2, base + 2] = gear[2]
            self.J[base + 2, base + 1] = gear[2] * hk
            motor_gear[base + 0] = gear[0]
            motor_gear[base + 1] = gear[1]
            motor_gear[base + 2] = gear[2]
            self.force_limit[base + 0] = 12.0
            self.force_limit[base + 1] = 12.0
            self.force_limit[base + 2] = 24.0

        self.JT = self.J.T.copy()
        motor_gear_sq = motor_gear ** 2
        self.motor_kp = 20.0 / motor_gear_sq
        self.motor_kd = 0.7 / motor_gear_sq

    def reset_history(self):
        """Initializes rolling observation history."""
        self.buf_cmd = [np.zeros(3, dtype=np.float32) for _ in range(self.hist_len)]
        self.buf_ang_vel = [np.zeros(3, dtype=np.float32) for _ in range(self.hist_len)]
        self.buf_proj_g = [np.array([0.0, 0.0, -1.0], dtype=np.float32) for _ in range(self.hist_len)]
        self.buf_jpos = [np.zeros(12, dtype=np.float32) for _ in range(self.hist_len)]
        self.buf_jvel = [np.zeros(12, dtype=np.float32) for _ in range(self.hist_len)]
        self.buf_actions = [np.zeros(12, dtype=np.float32) for _ in range(self.hist_len)]
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset_pose(self, x: float, y: float, z: float = 0.34, yaw: float = 0.0):
        """Sets robot root pose and resets leg joints to nominal stance."""
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = [x, y, z]
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = [qw, 0.0, 0.0, qz]
        self.data.qpos[self.joint_qpos_adrs] = DEFAULT_QPOS
        self.data.qvel[self.root_dof_adr : self.root_dof_adr + 6] = 0.0
        self.data.qvel[self.joint_dof_adrs] = 0.0
        self.q_target = DEFAULT_QPOS.copy()
        self.cmd_base[:] = 0.0
        self.target_cmd[:] = 0.0
        self.reset_history()

    @property
    def position(self) -> Tuple[float, float, float]:
        """Current (x, y, z) base position in world frame."""
        pos = self.data.xpos[self.base_body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    @property
    def yaw(self) -> float:
        """Current base yaw heading in radians."""
        # Derived directly from 3x3 world rotation matrix xmat: R00, R01, R02, R10, R11, R12, R20, R21, R22
        mat = self.data.xmat[self.base_body_id]
        return math.atan2(float(mat[3]), float(mat[0]))

    @property
    def is_fallen(self) -> bool:
        """Returns True if the quadruped has tipped over."""
        pos = self.position
        if pos[2] < -1.0:
            return False  # Intentionally placed underground / eliminated
        if pos[2] < 0.18:
            return True
        mat = self.data.xmat[self.base_body_id]
        return bool(mat[8] < 0.5)

    def recover_if_fallen(self) -> bool:
        """If quadruped has tipped over, cleanly upright it at its current location."""
        if self.is_fallen:
            pos = self.position
            cur_yaw = self.yaw
            qw = math.cos(cur_yaw / 2.0)
            qz = math.sin(cur_yaw / 2.0)
            self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = [pos[0], pos[1], 0.34]
            self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = [qw, 0.0, 0.0, qz]
            self.data.qpos[self.joint_qpos_adrs] = DEFAULT_QPOS
            self.data.qvel[self.root_dof_adr : self.root_dof_adr + 6] = 0.0
            self.data.qvel[self.joint_dof_adrs] = 0.0
            self.q_target = DEFAULT_QPOS.copy()
            self.cmd_base[:] = 0.0
            self.target_cmd[:] = 0.0
            self.reset_history()
            return True
        return False

    def set_world_velocity_target(self, vx_world: float, vy_world: float, speed_mult: float = 1.0):
        """Translates target 2D world velocities into agile, fluid omnidirectional quadruped locomotion.
        
        Projects world velocity targets directly into the robot's local body frame (vx_body, vy_body)
        with smooth, critically-damped yaw heading regulation and a deadband to eliminate ray noise twitching.
        """
        v_mag = math.hypot(vx_world, vy_world) * speed_mult
        if v_mag > 0.08:
            raw_target_yaw = math.atan2(vy_world, vx_world)
            if self.filtered_target_yaw is None:
                self.filtered_target_yaw = raw_target_yaw

            # Angular difference between raw velocity direction and current filtered heading
            err = math.atan2(math.sin(raw_target_yaw - self.filtered_target_yaw), math.cos(raw_target_yaw - self.filtered_target_yaw))

            # Deadband: ignore transient steering ray fluctuations (< 4.5 degrees = 0.08 rad)
            # This completely eliminates random twitching from discrete ray switching
            if abs(err) > 0.08:
                # Critically damped rotation rate: at most 2.0 rad/s (approx 115 deg/s) -> 0.04 rad per 50 Hz frame
                max_step = 2.0 * 0.02
                self.filtered_target_yaw += float(np.clip(err * 0.22, -max_step, max_step))

            # Current physical yaw and heading error relative to filtered target
            cur_yaw = self.yaw
            yaw_err = math.atan2(math.sin(self.filtered_target_yaw - cur_yaw), math.cos(self.filtered_target_yaw - cur_yaw))

            # Only command active yaw rotation if error exceeds deadband (prevents micro-oscillations)
            if abs(yaw_err) > 0.05:
                wz_target = float(np.clip(2.2 * yaw_err, -1.6, 1.6))
            else:
                wz_target = 0.0

            # Body-frame velocity projection relative to current heading:
            # Forward component along robot's nose:
            vb_x = (vx_world * math.cos(cur_yaw) + vy_world * math.sin(cur_yaw)) * speed_mult
            # Lateral component (positive left, negative right):
            vb_y = (-vx_world * math.sin(cur_yaw) + vy_world * math.cos(cur_yaw)) * speed_mult

            # Forward/reverse trotting bounded within RL policy stable training envelope [-0.2, 1.0]
            vx_target = float(np.clip(vb_x, -0.2, 1.0))

            # Lateral trotting bounded within stable envelope [-0.35, 0.35]
            vy_target = float(np.clip(vb_y, -0.35, 0.35))

            self.target_cmd[0] = vx_target
            self.target_cmd[1] = vy_target
            self.target_cmd[2] = wz_target
        else:
            self.target_cmd[:] = 0.0

    def step_policy(self):
        """Evaluates 50 Hz locomotion policy and updates joint position targets."""
        # 1. Acceleration / Slew-rate limiting on base command (responsive yet physically stable)
        dvx = float(np.clip(self.target_cmd[0] - self.cmd_base[0], -0.12, 0.12))
        dvy = float(np.clip(self.target_cmd[1] - self.cmd_base[1], -0.09, 0.09))
        dwz = float(np.clip(self.target_cmd[2] - self.cmd_base[2], -0.15, 0.15))
        self.cmd_base[0] = float(np.clip(self.cmd_base[0] + dvx, -0.2, 1.0))
        self.cmd_base[1] = float(np.clip(self.cmd_base[1] + dvy, -0.5, 0.5))
        self.cmd_base[2] = float(np.clip(self.cmd_base[2] + dwz, -2.0, 2.0))

        # 2. Read IMU angular velocity (scaled by 0.25)
        imu_ang_vel = self.data.sensordata[self.imu_sensor_adr : self.imu_sensor_adr + 3] * 0.25

        # Projected gravity in base frame (derived directly from base rotation matrix xmat)
        mat = self.data.xmat[self.base_body_id]
        proj_g = np.array([-mat[6], -mat[7], -mat[8]], dtype=np.float32)

        # Joint positions relative to default stance
        jpos = (self.data.qpos[self.joint_qpos_adrs] - DEFAULT_QPOS).astype(np.float32)

        # Joint velocities scaled by 0.05
        jvel = (self.data.qvel[self.joint_dof_adrs] * 0.05).astype(np.float32)

        # Push to FIFO history buffers (chronological order)
        self.buf_cmd.pop(0); self.buf_cmd.append(self.cmd_base.copy())
        self.buf_ang_vel.pop(0); self.buf_ang_vel.append(imu_ang_vel.astype(np.float32))
        self.buf_proj_g.pop(0); self.buf_proj_g.append(proj_g.astype(np.float32))
        self.buf_jpos.pop(0); self.buf_jpos.append(jpos)
        self.buf_jvel.pop(0); self.buf_jvel.append(jvel)
        self.buf_actions.pop(0); self.buf_actions.append(self.last_action.copy())

        # Assemble exact 270-dim term-major observation
        obs = np.concatenate([
            np.concatenate(self.buf_cmd),
            np.concatenate(self.buf_ang_vel),
            np.concatenate(self.buf_proj_g),
            np.concatenate(self.buf_jpos),
            np.concatenate(self.buf_jvel),
            np.concatenate(self.buf_actions)
        ])

        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

        self.last_action = action
        self.q_target = DEFAULT_QPOS + action * ACTION_SCALE

    def step_pd(self):
        """Computes 500 Hz parallel mechanism drivetrain torques and applies to actuators."""
        q = self.data.qpos[self.joint_qpos_adrs]
        v = self.data.qvel[self.joint_dof_adrs]

        motor_pos = q @ self.JT
        motor_pos_des = self.q_target @ self.JT
        motor_vel = v @ self.JT

        tau_motor = self.motor_kp * (motor_pos_des - motor_pos) - self.motor_kd * motor_vel
        tau_joint = tau_motor @ self.J

        ctrl = np.clip(tau_joint, -self.force_limit, self.force_limit)
        self.data.ctrl[self.actuator_adrs] = ctrl
