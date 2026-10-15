"""Manager terms for the native-axis BU10 wheel profile; no legacy body/joint assumptions."""
from __future__ import annotations

from dataclasses import MISSING
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from wheel_humanoid_lab.assets.q1_inertia import Q1_INERTIA_WHEEL_RADIUS


class InertiaMimicAction(ActionTerm):
    """Four passive servos with no policy channels, updated every physics substep."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._follower_ids, _ = self._asset.find_joints(cfg.follower_joints, preserve_order=True)
        self._driver_ids, _ = self._asset.find_joints(cfg.driver_joints, preserve_order=True)
        if len(self._follower_ids) != 4 or len(self._driver_ids) != 4:
            raise ValueError("Native Q1 inertia asset must expose all four mimic joints and drivers")
        self._multipliers = torch.tensor(cfg.multipliers, device=self.device).unsqueeze(0)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)
        self.position_target = self._asset.data.joint_pos[:, self._driver_ids] * self._multipliers

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty

    @property
    def processed_actions(self):
        return self._empty

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        self.position_target = self._asset.data.joint_pos[:, self._driver_ids] * self._multipliers
        velocity_target = self._asset.data.joint_vel[:, self._driver_ids] * self._multipliers
        self._asset.set_joint_position_target(self.position_target, joint_ids=self._follower_ids)
        self._asset.set_joint_velocity_target(velocity_target, joint_ids=self._follower_ids)

    def reset(self, env_ids=None):
        pass


@configclass
class InertiaMimicActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = InertiaMimicAction
    follower_joints: list[str] = MISSING
    driver_joints: list[str] = MISSING
    multipliers: list[float] = MISSING


def joint_pos_contract(env, asset_cfg: SceneEntityCfg):
    """Continuous wheel angle has no physical phase; retain its two zero observation slots."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    q = q.clone()
    q[:, -2:] = 0.0
    return q


def wheel_velocity_tracking(env, command_name: str, asset_cfg: SceneEntityCfg, std: float = 0.5):
    robot = env.scene[asset_cfg.name]
    # Both legs use native frames; left axle is -Y and right axle +Y.
    signs = torch.tensor([-1.0, 1.0], device=env.device)
    rim = robot.data.joint_vel[:, asset_cfg.joint_ids] * signs * Q1_INERTIA_WHEEL_RADIUS
    target = env.command_manager.get_command(command_name)[:, 0:1]
    return torch.exp(-torch.square(rim - target).mean(dim=1) / std**2)


def wheels_grounded(env, sensor_cfg: SceneEntityCfg, threshold: float = 5.0):
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).to(torch.float32).mean(dim=1)


def mimic_error_l2(env, asset_cfg: SceneEntityCfg, driver_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    multipliers = torch.tensor([0.444444, 0.444444, -1.0, -1.0], device=env.device)
    error = robot.data.joint_pos[:, asset_cfg.joint_ids] - multipliers * robot.data.joint_pos[:, driver_cfg.joint_ids]
    return torch.square(error).sum(dim=1)


def nonfinite_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return ~(torch.isfinite(robot.data.root_state_w).all(dim=1)
             & torch.isfinite(robot.data.joint_pos).all(dim=1)
             & torch.isfinite(robot.data.joint_vel).all(dim=1))
