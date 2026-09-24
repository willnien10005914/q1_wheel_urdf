"""Reward terms for the Q1 inline-skate / X2 swizzle task (Microduck roller recipe, adapted)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
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


# ------------------------------------------------------------------------------------------------
# posture task (kneel <-> stand) terms. ``command_name`` is the 1-D PostureCommand (1 = kneel).
# ------------------------------------------------------------------------------------------------
def _posture_target(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, 0]


def posture_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    stand_z: float,
    kneel_z: float,
    std: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian on pelvis height vs the height of the commanded posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    k = _posture_target(env, command_name)
    z_t = stand_z + (kneel_z - stand_z) * k
    return torch.exp(-((asset.data.root_pos_w[:, 2] - z_t) ** 2) / std**2)


class posture_keyframe(ManagerTermBase):
    """exp(-RMS(q - q_target)^2 / std^2) over the listed joints; q_target is the stand default pose or the
    kneel keyframe (``kneel_pose`` {joint regex: value}) depending on the posture command."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.ids = torch.tensor(asset_cfg.joint_ids, device=env.device)
        names = [self.asset.joint_names[i] for i in asset_cfg.joint_ids]
        kneel = self.asset.data.default_joint_pos[0, self.ids].clone()
        for i, n in enumerate(names):
            for pat, val in cfg.params["kneel_pose"].items():
                if re.fullmatch(pat, n):
                    kneel[i] = float(val)
        self.kneel = kneel.unsqueeze(0)

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, kneel_pose: dict, std: float = 0.3) -> torch.Tensor:
        k = _posture_target(env, command_name).unsqueeze(1)
        q_t = self.asset.data.default_joint_pos[:, self.ids] * (1.0 - k) + self.kneel * k
        dq = self.asset.data.joint_pos[:, self.ids] - q_t
        return torch.exp(-torch.mean(dq**2, dim=1) / std**2)


def posture_contacts(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_cfg: SceneEntityCfg,
    roller_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Kneel target: both wheels AND both rollers loaded. Stand target: both wheels loaded, rollers free."""
    sensor: ContactSensor = env.scene.sensors[wheel_cfg.name]
    f = sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]
    wheels = (f[:, wheel_cfg.body_ids] > threshold).all(dim=1).float()
    rollers = (f[:, roller_cfg.body_ids] > threshold).all(dim=1).float()
    rollers_free = (f[:, roller_cfg.body_ids] <= threshold).all(dim=1).float()
    k = _posture_target(env, command_name)
    return wheels * (k * rollers + (1.0 - k) * rollers_free)


def roller_contact_when_standing(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Roller touch is only a fault while the stand posture is commanded."""
    return any_contact(env, sensor_cfg, threshold) * (1.0 - _posture_target(env, command_name))


def stationary_base(env: ManagerBasedRLEnv, std: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Planar base speed near zero (the posture task must not roll away while squatting)."""
    v = env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(v**2, dim=1) / std**2)


def posture_still(
    env: ManagerBasedRLEnv,
    posture_command_name: str,
    velocity_command_name: str,
    std: float = 0.3,
    cmd_eps: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stay put while standing, and while kneeling with no drive command. A kneel drive command gates this off."""
    k = _posture_target(env, posture_command_name)
    cmd = env.command_manager.get_command(velocity_command_name)
    driving = ((cmd[:, 0].abs() > cmd_eps) | (cmd[:, 2].abs() > cmd_eps)).float()
    gate = (1.0 - k) + k * (1.0 - driving)
    return stationary_base(env, std, asset_cfg) * gate


def kneel_drive(
    env: ManagerBasedRLEnv,
    posture_command_name: str,
    velocity_command_name: str,
    std_vx: float = 0.25,
    std_yaw: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track body vx and yaw only while the kneel posture is commanded."""
    asset: Articulation = env.scene[asset_cfg.name]
    k = _posture_target(env, posture_command_name)
    cmd = env.command_manager.get_command(velocity_command_name)
    vx_err = cmd[:, 0] - asset.data.root_lin_vel_b[:, 0]
    yaw_err = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-(vx_err**2) / std_vx**2) * torch.exp(-(yaw_err**2) / std_yaw**2) * k


# ------------------------------------------------------------------------------------------------
# slide / push-skate terms
# ------------------------------------------------------------------------------------------------
def single_support(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 5.0,
    min_cmd: float = 0.2,
    min_air: float = 0.05,
    max_air: float = 0.22,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exactly one wheel loaded, and that unweight is still inside ``[min_air, max_air]`` s.

    Without the air-time window this term pays forever for a planted one-wheel glide (the failed
    ``slide_8192envs_lift`` habit). ``air_over_cap`` / ``long_air`` handle anything longer.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    one = ((f > threshold).sum(dim=1) == 1).float()
    in_window = ((air > min_air) & (air <= max_air)).any(dim=1).float()
    cmd = env.command_manager.get_command(command_name)
    return one * in_window * (cmd[:, 0] > min_cmd).float() * _upright_gate(env, asset, max_tilt, min_height)


def micro_unweight(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_air: float = 0.05,
    max_air: float = 0.22,
    load_thr: float = 5.0,
    min_cmd: float = 0.2,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense micro-lift: exactly one wheel airborne for ``[min_air, max_air]`` s, the other loaded.

    Drops to 0 the moment the lift exceeds ``max_air``, so a 0.4 s one-wheel glide does not score.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    one_air = ((air > min_air).sum(dim=1) == 1)
    in_window = ((air > min_air) & (air <= max_air)).any(dim=1)
    one_loaded = (f > load_thr).sum(dim=1) == 1
    cmd = env.command_manager.get_command(command_name)
    return (one_air & in_window & one_loaded).float() * (cmd[:, 0] > min_cmd).float() * _upright_gate(
        env, asset, max_tilt, min_height
    )


def air_over_cap(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, cap: float = 0.22) -> torch.Tensor:
    """Binary cost: any wheel has been off the ground longer than ``cap`` (default 220 ms)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return (air > cap).any(dim=1).float()


def one_wheel_glide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    load_thr: float = 5.0,
    unload_thr: float = 1.5,
    min_cmd: float = 0.2,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense X2 cue: one wheel clearly loaded, the other clearly unloaded, while moving upright."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    loaded = (f > load_thr).sum(dim=1) == 1
    unloaded = (f < unload_thr).sum(dim=1) == 1
    cmd = env.command_manager.get_command(command_name)
    return (loaded & unloaded).float() * (cmd[:, 0] > min_cmd).float() * _upright_gate(env, asset, max_tilt, min_height)


def double_support_when_moving(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 5.0,
    min_cmd: float = 0.25,
) -> torch.Tensor:
    """Cost: both wheels planted while a forward stride is commanded (the failed slide habit)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    both = (f > threshold).all(dim=1).float()
    cmd = env.command_manager.get_command(command_name)
    return both * (cmd[:, 0] > min_cmd).float()


def stride_leg_split(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.30,
    min_cmd: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pay for L/R hip-pitch and knee difference while moving (a push stride, not a symmetric stance)."""
    asset: Articulation = env.scene[asset_cfg.name]
    names = asset.joint_names
    idx = {n: i for i, n in enumerate(names)}
    dq = (asset.data.joint_pos[:, idx["l_hip_pitch_joint"]] - asset.data.joint_pos[:, idx["r_hip_pitch_joint"]]).abs()
    dq = dq + (asset.data.joint_pos[:, idx["l_knee_joint"]] - asset.data.joint_pos[:, idx["r_knee_joint"]]).abs()
    cmd = env.command_manager.get_command(command_name)
    return (1.0 - torch.exp(-dq / std)) * (cmd[:, 0] > min_cmd).float()


class stride_alternation(ManagerTermBase):
    """Dense: pay while the currently airborne wheel is the opposite of the last completed lift.

    The previous pulse-on-touchdown version was ~1 bonus per stride vs thousands of per-step
    single-support points, so the policy parked on one wheel. Completing a lift (air → 0) records
    that side; the next lift only scores if it is the other wheel.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self.n_prev = torch.zeros(n, dtype=torch.long, device=env.device)
        self.cur = torch.zeros(n, dtype=torch.long, device=env.device)
        self.completed = torch.full((n,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.n_prev[:] = 0
            self.cur[:] = 0
            self.completed[:] = -1
        else:
            self.n_prev[env_ids] = 0
            self.cur[env_ids] = 0
            self.completed[env_ids] = -1

    def __call__(
        self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, min_air: float = 0.05
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
        n = (air > min_air).sum(dim=1)
        side = torch.argmax(air, dim=1)
        ended = (self.n_prev == 1) & (n == 0)
        self.completed = torch.where(ended, self.cur, self.completed)
        self.cur = torch.where(n == 1, side, self.cur)
        alt = (n == 1) & (self.completed >= 0) & (side != self.completed)
        self.n_prev = n.long()
        cmd = env.command_manager.get_command(command_name)
        return alt.float() * (cmd[:, 0] > 0.2).float()


def _hip_pitch_ids(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    key = "_q1_hip_pitch_lr"
    cache = getattr(env, key, None)
    if cache is None:
        ids, _ = asset.find_joints(["l_hip_pitch_joint", "r_hip_pitch_joint"], preserve_order=True)
        cache = torch.tensor(ids, device=env.device)
        setattr(env, key, cache)
    return cache


def loaded_wheel_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    wheel_radius: float = WHEEL_RADIUS,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Rim speed of the wheels that are on the ground must match vx_cmd.

    An airborne wheel is ignored, so a micro-lift does not spoil the speed match.
    ``asset_cfg`` joints and ``sensor_cfg`` bodies must be the same left-then-right order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    rim = wheel_radius * asset.data.joint_vel[:, asset_cfg.joint_ids]
    loaded = (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > threshold).float()
    n = loaded.sum(dim=1).clamp(min=1.0)
    mean_rim = (rim * loaded).sum(dim=1) / n
    err = mean_rim - cmd[:, 0]
    return torch.exp(-(err**2) / std**2) * (loaded.sum(dim=1) > 0).float()


class rear_foot_unweight(ManagerTermBase):
    """X2 push: the foot that is farther back unweights, briefly.

    Hip pitch about +Y swings the thigh backward as the angle increases, so the larger
    pitch is the rear foot. A frozen weight shift scores nothing: the rear force has to
    be falling, or that wheel has to be in the air for at most ``max_air`` seconds.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self.prev_rear = torch.full((n,), -1.0, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.prev_rear[:] = -1.0
        else:
            self.prev_rear[env_ids] = -1.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        min_split: float = 0.28,
        min_air: float = 0.02,
        max_air: float = 0.22,
        min_cmd: float = 0.25,
        max_tilt: float = 0.45,
        min_height: float = 0.65,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        ids = _hip_pitch_ids(env, asset)
        split = asset.data.joint_pos[:, ids[0]] - asset.data.joint_pos[:, ids[1]]
        left_rear = split > 0
        f = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
        air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
        f_rear = torch.where(left_rear, f[:, 0], f[:, 1])
        f_front = torch.where(left_rear, f[:, 1], f[:, 0])
        air_rear = torch.where(left_rear, air[:, 0], air[:, 1])
        falling = (self.prev_rear >= 0) & (f_rear < self.prev_rear - 1.0)
        in_air = (air_rear > min_air) & (air_rear <= max_air)
        self.prev_rear = f_rear
        shift = torch.tanh((f_front - f_rear) / 25.0).clamp(min=0.0)
        cmd = env.command_manager.get_command(command_name)
        gate = (split.abs() > min_split).float() * (cmd[:, 0] > min_cmd).float()
        gate = gate * _upright_gate(env, asset, max_tilt, min_height)
        return shift * (falling | in_air).float() * (air_rear <= max_air).float() * gate


class stride_swap(ManagerTermBase):
    """Pay only while the front/back stride is actually swapping feet.

    ``l_hip_pitch - r_hip_pitch`` changes sign when the other foot becomes the rear one.
    A static split (one leg parked forward) never swaps, so it scores 0.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        dev = env.device
        self.prev = torch.zeros(n, device=dev)
        self.last_sign = torch.zeros(n, device=dev)
        self.since = torch.full((n,), 10.0, device=dev)
        self.have = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.prev[:] = 0
            self.last_sign[:] = 0
            self.since[:] = 10.0
            self.have[:] = False
        else:
            self.prev[env_ids] = 0
            self.last_sign[env_ids] = 0
            self.since[env_ids] = 10.0
            self.have[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        min_split: float = 0.30,
        swap_window: float = 0.9,
        min_cmd: float = 0.25,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        ids = _hip_pitch_ids(env, asset)
        d = asset.data.joint_pos[:, ids[0]] - asset.data.joint_pos[:, ids[1]]
        sign = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
        strong = d.abs() > min_split
        swapped = strong & self.have & (sign != self.last_sign) & (self.last_sign != 0)
        self.since = torch.where(swapped, torch.zeros_like(self.since), self.since + env.step_dt)
        self.last_sign = torch.where(strong, sign, self.last_sign)
        self.have = self.have | strong
        self.prev = d
        depth = 1.0 - torch.exp(-d.abs() / 0.45)
        cmd = env.command_manager.get_command(command_name)
        recent = (self.since < swap_window).float()
        return recent * depth * (cmd[:, 0] > min_cmd).float()


def roll_shift(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    lo: float = 0.02,
    hi: float = 0.10,
    min_air: float = 0.02,
    max_air: float = 0.22,
    min_cmd: float = 0.25,
    max_tilt: float = 0.45,
    min_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small left/right lean while exactly one wheel is in the micro-lift window."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    roll = asset.data.projected_gravity_b[:, 1].abs()
    band = ((roll > lo) & (roll < hi)).float()
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    one = ((air > min_air) & (air <= max_air)).sum(dim=1) == 1
    cmd = env.command_manager.get_command(command_name)
    return band * one.float() * (cmd[:, 0] > min_cmd).float() * _upright_gate(env, asset, max_tilt, min_height)
