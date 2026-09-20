"""Reward terms for the Q1 inline-skate / X2 swizzle task (Microduck roller recipe, adapted)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from wheel_humanoid_lab.assets import WHEEL_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _upright_gate(env: ManagerBasedRLEnv, asset: Articulation, max_tilt: float, min_height: float) -> torch.Tensor:
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
    return ((tilt < max_tilt) & (asset.data.root_pos_w[:, 2] > min_height)).float()


def _wheel_shin_ids(env: ManagerBasedRLEnv, asset: Articulation, wheel_body_ids) -> tuple[torch.Tensor, torch.Tensor]:
    key = "_q1_wheel_shin_ids"
    cache = getattr(env, key, None)
    if cache is None:
        wheel_names = [asset.body_names[i] for i in wheel_body_ids]
        shin_names = [n.replace("_wheel_link", "_shin_link") for n in wheel_names]
        shin_ids, _ = asset.find_bodies(shin_names, preserve_order=True)
        cache = (torch.tensor(list(wheel_body_ids), device=env.device), torch.tensor(shin_ids, device=env.device))
        setattr(env, key, cache)
    return cache


def wheel_ground_velocities(
    env: ManagerBasedRLEnv, asset: Articulation, wheel_body_ids, wheel_joint_ids, wheel_radius: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (rim_speed r*omega, longitudinal hub speed, lateral hub speed) per wheel, shapes (N, W)."""
    w_ids, s_ids = _wheel_shin_ids(env, asset, wheel_body_ids)
    v_hub = asset.data.body_lin_vel_w[:, w_ids]  # (N, W, 3)
    q_shin = asset.data.body_quat_w[:, s_ids]  # (N, W, 4)
    n, w = v_hub.shape[:2]
    fwd = math_utils.quat_apply(q_shin.reshape(-1, 4), torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(n * w, 3))
    left = math_utils.quat_apply(q_shin.reshape(-1, 4), torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(n * w, 3))
    fwd = fwd.reshape(n, w, 3)
    left = left.reshape(n, w, 3)
    fwd[..., 2] = 0.0
    left[..., 2] = 0.0
    fwd = torch.nn.functional.normalize(fwd, dim=-1)
    left = torch.nn.functional.normalize(left, dim=-1)
    v_long = (v_hub * fwd).sum(-1)
    v_lat = (v_hub * left).sum(-1)
    rim = wheel_radius * asset.data.joint_vel[:, wheel_joint_ids]
    return rim, v_long, v_lat


# ------------------------------------------------------------------------------------------------
# positive task terms
# ------------------------------------------------------------------------------------------------
def wheel_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = WHEEL_RADIUS,
) -> torch.Tensor:
    """Mean wheel rim speed r*omega must match vx_cmd (the CubeMars wheel motors have to spin)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    rim = wheel_radius * asset.data.joint_vel[:, asset_cfg.joint_ids]
    err = rim.mean(dim=1) - cmd[:, 0]
    return torch.exp(-(err**2) / std**2)


def base_vx_track(env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Planar base velocity tracking (vx, vy) in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-err / std**2)


def leg_symmetry(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, pairs: list, std: float) -> torch.Tensor:
    """exp(-L1/std) of mirrored joint pairs; ``pairs`` = [(left, right, sign)] with q_L - sign*q_R = 0."""
    asset: Articulation = env.scene[asset_cfg.name]
    key = "_q1_sym_" + "_".join(p[0] for p in pairs)
    cache = getattr(env, key, None)
    if cache is None:
        l_ids, _ = asset.find_joints([p[0] for p in pairs], preserve_order=True)
        r_ids, _ = asset.find_joints([p[1] for p in pairs], preserve_order=True)
        cache = (
            torch.tensor(l_ids, device=env.device),
            torch.tensor(r_ids, device=env.device),
            torch.tensor([p[2] for p in pairs], device=env.device),
        )
        setattr(env, key, cache)
    l_ids, r_ids, sign = cache
    q = asset.data.joint_pos
    l1 = torch.abs(q[:, l_ids] - sign * q[:, r_ids]).sum(dim=1)
    return torch.exp(-l1 / std)


def grounded(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, threshold: float = 5.0) -> torch.Tensor:
    """Both wheels loaded while vx_cmd >= 0 (double support is the default skate stance)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    both = (f > threshold).all(dim=1).float()
    cmd = env.command_manager.get_command(command_name)
    return both * (cmd[:, 0] >= 0.0).float()


def skating_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_air: float = 0.05,
    max_air: float = 0.25,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Micro-lift: on touchdown, reward an unweighting of [min_air, max_air] s. Longer lifts pay nothing
    extra here and are punished by ``long_air``. Gated on upright + a non-zero speed command."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    r = (torch.clamp(last_air, max=max_air) - min_air).clamp(min=0.0) * first_contact.float()
    r = r.sum(dim=1)
    cmd = env.command_manager.get_command(command_name)
    return r * (cmd[:, 0].abs() > 0.1).float() * _upright_gate(env, asset, max_tilt, min_height)


def forward_lean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float = 0.17,
    std: float = 0.12,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
) -> torch.Tensor:
    """Gaussian on the sagittal tilt of ``asset_cfg.body_ids[0]`` (torso): projected_gravity_b[0] near
    ``target`` (+ = CoM ahead of the axles). Zero when the trunk tilt exceeds ``max_tilt`` or the base is
    low, so a face-plant never pays as lean."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    g_b = math_utils.quat_apply_inverse(q, asset.data.GRAVITY_VEC_W)
    r = torch.exp(-((g_b[:, 0] - target) ** 2) / std**2)
    tilt = torch.acos(torch.clamp(-g_b[:, 2], -1.0, 1.0))
    gate = (tilt < max_tilt).float() * (asset.data.root_pos_w[:, 2] > min_height).float()
    return r * gate


def arms_back(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 0.45,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
) -> torch.Tensor:
    """Gaussian to the arms-back skate keyframe (the articulation default pose on the listed joints).
    Gated on upright. Not a pull to HOME: the keyframe is the X2 style pose.

    Uses the per-joint RMS error so ``std`` is a per-joint angle (rad) and the term keeps a gradient
    for 6-8 joints; a sum over joints with a narrow std is ~0 everywhere except at the keyframe."""
    asset: Articulation = env.scene[asset_cfg.name]
    dq = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    r = torch.exp(-torch.mean(dq**2, dim=1) / std**2)
    return r * _upright_gate(env, asset, max_tilt, min_height)


def upright(env: ManagerBasedRLEnv, std: float = 0.35, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.exp(-torch.sum(g[:, :2] ** 2, dim=1) / std**2)


def head_up(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.3) -> torch.Tensor:
    """Head/neck link roughly world-up."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    g_b = math_utils.quat_apply_inverse(q, asset.data.GRAVITY_VEC_W)
    return torch.exp(-torch.sum(g_b[:, :2] ** 2, dim=1) / std**2)


# ------------------------------------------------------------------------------------------------
# costs
# ------------------------------------------------------------------------------------------------
def wheel_slip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    wheel_radius: float = WHEEL_RADIUS,
    threshold: float = 5.0,
) -> torch.Tensor:
    """|r*omega - v_hub_longitudinal| summed over loaded wheels (sliding the base without spinning the
    motors is a cheat)."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    rim, v_long, _ = wheel_ground_velocities(env, asset, sensor_cfg.body_ids, asset_cfg.joint_ids, wheel_radius)
    loaded = (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > threshold).float()
    return (torch.abs(rim - v_long) * loaded).sum(dim=1)


def wheel_lateral_slip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    wheel_radius: float = WHEEL_RADIUS,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Sideways hub velocity of loaded wheels (an inline wheel cannot roll sideways)."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    _, _, v_lat = wheel_ground_velocities(env, asset, sensor_cfg.body_ids, asset_cfg.joint_ids, wheel_radius)
    loaded = (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > threshold).float()
    return (torch.abs(v_lat) * loaded).sum(dim=1)


def flying(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air: float = 0.08) -> torch.Tensor:
    """Both wheels off the ground for more than ``max_air`` s."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return (air > max_air).all(dim=1).float()


def long_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, cap: float = 0.30) -> torch.Tensor:
    """Excess single-wheel air time beyond ``cap`` (a full step is worse than a micro-unweight)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return (air - cap).clamp(min=0.0).sum(dim=1)


def ang_vel_roll_yaw_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll and yaw rate of the base (pitch rate is allowed while the lean settles)."""
    w = env.scene[asset_cfg.name].data.root_ang_vel_b
    return w[:, 0] ** 2 + w[:, 2] ** 2


def home_pose_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 pull to the default pose. Only for unused joints (neck, wrists, grippers, waist yaw/roll)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def actuator_saturation(env: ManagerBasedRLEnv, actuator_names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Fraction of joints in the listed CubeMars groups that hit the current/torque limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    sat = [asset.actuators[n].saturated.float() for n in actuator_names if hasattr(asset.actuators[n], "saturated")]
    if not sat:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.cat(sat, dim=1).mean(dim=1)


def any_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """1 when any listed body is in contact (used as a penalty for the knee rollers)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (f > threshold).any(dim=1).float()
