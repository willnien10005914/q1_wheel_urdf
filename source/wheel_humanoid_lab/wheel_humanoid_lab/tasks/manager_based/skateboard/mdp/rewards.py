"""Reward helpers for dual-wheel standing skate (AgiBot X2 style)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def both_contacts(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Reward 1 when every listed body is in contact, else the contact fraction."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(net_forces, dim=-1) > threshold
    return torch.mean(in_contact.float(), dim=1)


def lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded planar velocity in the robot base frame."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def skate_pose_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize deviation from the default (skateboard) joint pose, excluding wheels."""
    asset = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(current_pos - default_pos), dim=1)


def wheel_speed_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Small penalty on wheel speed so the policy does not spin unloaded wheels."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def wheel_roll_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float = 0.10,
    std: float = 4.0,
    wheel_sign: float = -1.0,
    track_half_width: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["l_wheel_joint", "r_wheel_joint"]),
) -> torch.Tensor:
    """Match wheel spin to commanded planar velocity (X2-style rolling skate).

    Wheel joints rotate about +Y. Positive spin moves the contact patch +X, so the
    robot translates -X. ``wheel_sign=-1`` converts body vx into that joint velocity.
    Left/right wheels use a differential term so yaw commands also roll the right way.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    vx = command[:, 0]
    wz = command[:, 2]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # joint_names resolve as [l_wheel_joint, r_wheel_joint] (alphabetical).
    y_off = torch.tensor([track_half_width, -track_half_width], device=vx.device, dtype=vx.dtype)
    body_speed = vx.unsqueeze(1) - wz.unsqueeze(1) * y_off.unsqueeze(0)
    target = wheel_sign * body_speed / wheel_radius
    err = torch.sum(torch.square(wheel_vel - target), dim=1)
    return torch.exp(-err / std**2)


def forward_lean(
    env: ManagerBasedRLEnv,
    command_name: str,
    gain: float = 0.10,
    std: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Lean a little into the commanded forward speed, like an inverted-pendulum skate."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_gx = gain * command[:, 0]
    err = torch.square(asset.data.projected_gravity_b[:, 0] - target_gx)
    return torch.exp(-err / std**2)
