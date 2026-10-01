"""Observation terms for the Q1 skate task.

Sensor realism lives here, not in the policy: IMU (Xsens MTi-630 on the pelvis) delay + mounting tilt,
encoder bias, one-step joint velocity delay. All per-env randomization is restore-then-sample on reset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import Imu
from isaaclab.utils import DelayBuffer

from wheel_humanoid_lab.assets import (
    CMD_ARM_STYLE_DIM,
    CMD_BODY_DIM,
    CMD_HEAD_DIM,
    CMD_TWIST_DIM,
    Q1_WHEEL_JOINTS,
    WHEEL_RADIUS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ids(env_ids, num_envs: int, device: str) -> torch.Tensor:
    if env_ids is None or env_ids == slice(None):
        return torch.arange(num_envs, device=device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device)
    return torch.as_tensor(list(env_ids), device=device)


def _small_random_quat(n: int, max_deg: float, device: str) -> torch.Tensor:
    """Random rotation of at most ``max_deg`` about a random axis, (w, x, y, z)."""
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=1)
    ang = torch.empty(n, device=device).uniform_(-math.radians(max_deg), math.radians(max_deg))
    return math_utils.quat_from_angle_axis(ang, axis)


class ImuObs(ManagerTermBase):
    """IMU gyro or projected gravity with per-env mounting tilt and a random output delay.

    params:
        sensor_cfg: SceneEntityCfg of the ``Imu`` sensor.
        quantity: ``"ang_vel"`` or ``"projected_gravity"``.
        max_delay: maximum delay in policy steps (resampled 0..max_delay on reset).
        tilt_deg: mounting misalignment DR, +- degrees about a random axis.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor: Imu = env.scene[cfg.params["sensor_cfg"].name]
        self.quantity: str = cfg.params.get("quantity", "ang_vel")
        self.max_delay: int = int(cfg.params.get("max_delay", 0))
        self.tilt_deg: float = float(cfg.params.get("tilt_deg", 0.0))
        self.buffer = DelayBuffer(self.max_delay, self.num_envs, device=self.device)
        self.tilt = torch.zeros(self.num_envs, 4, device=self.device)
        self.tilt[:, 0] = 1.0
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _ids(env_ids, self.num_envs, self.device)
        n = ids.numel()
        if self.tilt_deg > 0.0:
            self.tilt[ids] = _small_random_quat(n, self.tilt_deg, self.device)
        else:
            self.tilt[ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        if self.max_delay > 0:
            lags = torch.randint(0, self.max_delay + 1, (n,), dtype=torch.int, device=self.device)
            self.buffer.set_time_lag(lags, ids)
            self.buffer.reset(ids)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, quantity: str = "ang_vel", max_delay: int = 0, tilt_deg: float = 0.0) -> torch.Tensor:
        if quantity == "ang_vel":
            v = self.sensor.data.ang_vel_b
        else:
            v = self.sensor.data.projected_gravity_b
        v = math_utils.quat_apply(self.tilt, v)
        if self.max_delay > 0:
            v = self.buffer.compute(v)
        return v.clone()


class JointPosContract(ManagerTermBase):
    """``joint_pos - default`` over the 24-D contract order with a per-env encoder bias.

    Wheel joints are continuous: their angle is wrapped to [-pi, pi) so the observation stays bounded
    (the real AK10-9 output encoder is absolute anyway).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.ids, self.names = self.asset.find_joints(asset_cfg.joint_names, preserve_order=True)
        self.ids = torch.tensor(self.ids, device=self.device)
        self.wrap_mask = torch.tensor([n in Q1_WHEEL_JOINTS for n in self.names], device=self.device)
        self.bias_deg: float = float(cfg.params.get("bias_deg", 0.0))
        self.bias = torch.zeros(self.num_envs, len(self.names), device=self.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _ids(env_ids, self.num_envs, self.device)
        b = math.radians(self.bias_deg)
        self.bias[ids] = torch.empty(ids.numel(), self.bias.shape[1], device=self.device).uniform_(-b, b)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, bias_deg: float = 0.0) -> torch.Tensor:
        q = self.asset.data.joint_pos[:, self.ids] - self.asset.data.default_joint_pos[:, self.ids]
        q = torch.where(self.wrap_mask, math_utils.wrap_to_pi(q), q)
        return q + self.bias


class JointVelContract(ManagerTermBase):
    """Joint velocity over the 24-D contract order, delayed by a fixed number of policy steps."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        ids, _ = self.asset.find_joints(asset_cfg.joint_names, preserve_order=True)
        self.ids = torch.tensor(ids, device=self.device)
        self.delay: int = int(cfg.params.get("delay", 0))
        self.buffer = DelayBuffer(self.delay, self.num_envs, device=self.device)
        if self.delay > 0:
            self.buffer.set_time_lag(self.delay)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self.delay > 0:
            self.buffer.reset(_ids(env_ids, self.num_envs, self.device))

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, delay: int = 0) -> torch.Tensor:
        v = self.asset.data.joint_vel[:, self.ids]
        if self.delay > 0:
            v = self.buffer.compute(v)
        return v.clone()


def joint_pos_contract_clean(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged (critic) joint positions in contract order, wheels wrapped, no bias."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    mask = torch.tensor([n in Q1_WHEEL_JOINTS for n in names], device=env.device)
    return torch.where(mask, math_utils.wrap_to_pi(q), q)


def skate_command(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    arm_style: float = 1.0,
    posture_command_name: str | None = None,
) -> torch.Tensor:
    """Fixed-width command block: twist(3) | head(4) | body(6) | arm_style(1). Unused slots are zero.

    ``body[0]`` is the posture target (0 = standing skate, 1 = four-wheel kneel) when a posture command
    term is given (posture task); the skate/slide policies always see 0 there.
    """
    twist = env.command_manager.get_command(command_name)[:, :CMD_TWIST_DIM]
    n = twist.shape[0]
    head = torch.zeros(n, CMD_HEAD_DIM, device=env.device)
    body = torch.zeros(n, CMD_BODY_DIM, device=env.device)
    if posture_command_name is not None:
        body[:, 0] = env.command_manager.get_command(posture_command_name)[:, 0]
    style = torch.full((n, CMD_ARM_STYLE_DIM), float(arm_style), device=env.device)
    return torch.cat([twist, head, body, style], dim=1)


# Index of body[0] inside the 92-D actor observation (used by play.py to inject the posture target).
POSTURE_OBS_INDEX = 3 + 3 + 24 + 24 + 24 + CMD_TWIST_DIM + CMD_HEAD_DIM
# body[1]: get-up phase in [0, 1]. Skate and posture leave it at 0.
GETUP_PHASE_INDEX = POSTURE_OBS_INDEX + 1


def getup_command(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    arm_style: float = 1.0,
    phase_s: float = 8.0,
) -> torch.Tensor:
    """Skate command block with body[1] = episode phase so the MLP can see the get-up schedule."""
    cmd = skate_command(env, command_name=command_name, arm_style=arm_style)
    phase = (env.episode_length_buf.to(cmd.dtype) * env.step_dt / phase_s).clamp(0.0, 1.0)
    cmd[:, CMD_TWIST_DIM + CMD_HEAD_DIM + 1] = phase
    return cmd


def wheel_rim_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, wheel_radius: float = WHEEL_RADIUS) -> torch.Tensor:
    """r * omega for the wheel joints (privileged / debugging)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return wheel_radius * asset.data.joint_vel[:, asset_cfg.joint_ids]


def wheel_contact_normal_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, scale: float = 400.0) -> torch.Tensor:
    """Normal contact force on the wheel bodies divided by ``scale`` (privileged)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return f / scale


def wheel_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time[:, sensor_cfg.body_ids]
