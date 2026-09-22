"""Zero-width action term that enforces the knee-roller mimic linkage (roller = 0.444 * knee).

The URDF ``<mimic>`` tag is dropped by the USD importer, so without this the passive roller joints
flop to a limit under load and the four-wheel kneel geometry differs from the real robot.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RollerMimicAction(ActionTerm):
    cfg: RollerMimicActionCfg
    _asset: Articulation

    def __init__(self, cfg: RollerMimicActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._roller_ids, _ = self._asset.find_joints(cfg.roller_joint_names, preserve_order=True)
        self._knee_ids, _ = self._asset.find_joints(cfg.knee_joint_names, preserve_order=True)
        assert len(self._roller_ids) == len(self._knee_ids)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        knee = self._asset.data.joint_pos[:, self._knee_ids]
        self._asset.set_joint_position_target(self.cfg.multiplier * knee + self.cfg.offset, joint_ids=self._roller_ids)
        self._asset.set_joint_velocity_target(
            self.cfg.multiplier * self._asset.data.joint_vel[:, self._knee_ids], joint_ids=self._roller_ids
        )

    def reset(self, env_ids=None) -> None:
        pass


@configclass
class RollerMimicActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = RollerMimicAction
    roller_joint_names: list[str] = MISSING
    knee_joint_names: list[str] = MISSING
    multiplier: float = 0.444444
    offset: float = 0.0
