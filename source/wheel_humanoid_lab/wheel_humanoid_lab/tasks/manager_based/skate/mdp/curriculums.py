"""Stage curriculum + debug metrics for the Q1 skate task.

Stages (iteration based, see ``SKATE_STAGES`` in the env cfg):
  0 stand/balance on two wheels           3 forward lean + arms back
  1 wheel-driven glide, both wheels down  4 micro-lift allowed, vx up to 1.2-1.8 m/s
  2 swizzle at 0.3-0.8 m/s                5 gentle pushes, CoM/friction DR up, heading
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from wheel_humanoid_lab.assets import WHEEL_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class skate_stage(ManagerTermBase):
    """Applies the stage table to reward weights, command ranges, reset velocity, pushes and CoM DR.

    params:
        stages: list of dicts, each with keys
            ``iter`` (start iteration), ``rewards`` {term: weight}, ``cmd_vx`` (lo, hi), ``cmd_yaw`` (lo, hi),
            ``rel_standing`` float, ``reset_vx`` (lo, hi), ``push`` (lo, hi) | None, ``com`` float (m).
        steps_per_iter: env steps per PPO iteration (``num_steps_per_env``).
        start_iter: iteration offset (use when resuming a run in a later stage).
        force_stage: int | None, pin the stage (play / debugging).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.stages: list[dict] = cfg.params["stages"]
        self.steps_per_iter: int = int(cfg.params.get("steps_per_iter", 24))
        self.start_iter: int = int(cfg.params.get("start_iter", 0))
        self.force_stage = cfg.params.get("force_stage", None)
        self._applied: int = -1

    def _apply(self, env: ManagerBasedRLEnv, k: int):
        # stage dicts are deltas: replay 0..k so that jumping straight to a late stage (resume,
        # force_stage) reproduces the same settings as walking through the curriculum.
        for i in range(k + 1):
            self._apply_one(env, self.stages[i])
        self._applied = k

    def _apply_one(self, env: ManagerBasedRLEnv, st: dict):
        for name, w in st.get("rewards", {}).items():
            cfg = env.reward_manager.get_term_cfg(name)
            cfg.weight = float(w)
            env.reward_manager.set_term_cfg(name, cfg)
        cmd = env.command_manager.get_term("base_velocity").cfg
        if "cmd_vx" in st:
            cmd.ranges.lin_vel_x = tuple(st["cmd_vx"])
        if "cmd_yaw" in st:
            cmd.ranges.ang_vel_z = tuple(st["cmd_yaw"])
        if "rel_standing" in st:
            cmd.rel_standing_envs = float(st["rel_standing"])
        if "reset_vx" in st:
            env.event_manager.get_term_cfg("reset_base").params["velocity_range"]["x"] = tuple(st["reset_vx"])
        if "push" in st and "push_robot" in env.event_manager.active_terms.get("interval", []):
            p = st["push"]
            rng = {"x": (0.0, 0.0), "y": (0.0, 0.0)} if p is None else {"x": tuple(p), "y": tuple(p)}
            env.event_manager.get_term_cfg("push_robot").params["velocity_range"] = rng
        if "com" in st and "randomize_com" in env.event_manager.active_terms.get("reset", []):
            c = float(st["com"])
            env.event_manager.get_term_cfg("randomize_com").params["com_range"] = {"x": (-c, c), "y": (-c, c), "z": (-c, c)}

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], stages, steps_per_iter: int = 24, start_iter: int = 0, force_stage=None) -> float:
        if self.force_stage is not None:
            k = int(self.force_stage)
        else:
            it = self.start_iter + env.common_step_counter // self.steps_per_iter
            k = 0
            for i, st in enumerate(self.stages):
                if it >= int(st.get("iter", 0)):
                    k = i
        if k != self._applied:
            self._apply(env, k)
        return float(k)


class action_rate_schedule(ManagerTermBase):
    """Piecewise schedule of the ``action_rate_l2`` weight: (iteration, weight) knots, linear in between."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.knots: list[tuple[int, float]] = [tuple(k) for k in cfg.params["knots"]]
        self.steps_per_iter: int = int(cfg.params.get("steps_per_iter", 24))
        self.start_iter: int = int(cfg.params.get("start_iter", 0))
        self.term: str = cfg.params.get("term_name", "action_rate_l2")

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], knots, steps_per_iter: int = 24, start_iter: int = 0, term_name: str = "action_rate_l2") -> float:
        it = self.start_iter + env.common_step_counter // self.steps_per_iter
        w = self.knots[0][1]
        for (i0, w0), (i1, w1) in zip(self.knots[:-1], self.knots[1:]):
            if it >= i1:
                w = w1
            elif it >= i0:
                w = w0 + (w1 - w0) * (it - i0) / max(i1 - i0, 1)
                break
        cfg = env.reward_manager.get_term_cfg(self.term)
        if cfg.weight != w:
            cfg.weight = float(w)
            env.reward_manager.set_term_cfg(self.term, cfg)
        return float(w)


def skate_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg,
    shoulder_cfg: SceneEntityCfg,
    ake90_groups: list[str],
    wheel_radius: float = WHEEL_RADIUS,
) -> dict[str, float]:
    """Debug metrics over all envs (logged under Curriculum/): wheel omega, slip, contact fractions,
    peak air time, torso pitch, shoulder pitch, AKE90 saturation %."""
    from .rewards import _wheel_fore_aft, wheel_ground_velocities

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    rim, v_long, v_lat = wheel_ground_velocities(env, asset, sensor_cfg.body_ids, asset_cfg.joint_ids, wheel_radius)
    loaded = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 5.0
    n_loaded = loaded.sum(dim=1)
    g_t = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, torso_cfg.body_ids[0]], asset.data.GRAVITY_VEC_W)
    sat = [asset.actuators[n].saturated.float() for n in ake90_groups if hasattr(asset.actuators[n], "saturated")]
    sat_frac = torch.cat(sat, dim=1).mean() if sat else torch.zeros((), device=env.device)
    return {
        "wheel_omega_mean": float(asset.data.joint_vel[:, asset_cfg.joint_ids].mean()),
        "wheel_rim_speed_mean": float(rim.mean()),
        "wheel_slip_long": float((torch.abs(rim - v_long) * loaded.float()).sum(1).mean()),
        "wheel_slip_lat": float((torch.abs(v_lat) * loaded.float()).sum(1).mean()),
        "both_contact_frac": float((n_loaded == 2).float().mean()),
        "single_contact_frac": float((n_loaded == 1).float().mean()),
        "peak_air_time": float(sensor.data.current_air_time[:, sensor_cfg.body_ids].max()),
        "mean_air_time": float(sensor.data.current_air_time[:, sensor_cfg.body_ids].mean()),
        "air_over_022_frac": float((sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.22).any(dim=1).float().mean()),
        "torso_pitch_gx": float(g_t[:, 0].mean()),
        "shoulder_pitch_mean": float(asset.data.joint_pos[:, shoulder_cfg.joint_ids].mean()),
        "ake90_saturation_pct": float(100.0 * sat_frac),
        "base_vx_mean": float(asset.data.root_lin_vel_b[:, 0].mean()),
        "base_height_mean": float(asset.data.root_pos_w[:, 2].mean()),
        "fore_aft_abs_mean": float(_wheel_fore_aft(env, asset).abs().mean()),
    }
