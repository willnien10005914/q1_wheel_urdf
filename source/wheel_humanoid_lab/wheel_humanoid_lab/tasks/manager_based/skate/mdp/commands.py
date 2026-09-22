"""Posture command: a 1-D target (0 = standing skate pose, 1 = four-wheel kneel) that flips at random."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PostureCommand(CommandTerm):
    """Bernoulli posture target held for ``resampling_time_range`` seconds.

    The command is the body[0] slot of the observation command block, so the same 92-D contract
    (and ONNX export) is shared with the skate policies; they simply see body[0] == 0.
    """

    cfg: PostureCommandCfg

    def __init__(self, cfg: PostureCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.target = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["kneel_frac"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target

    def _update_metrics(self):
        self.metrics["kneel_frac"] = self.target[:, 0]

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        kneel = (torch.rand(n, device=self.device) < self.cfg.p_kneel).float()
        self.target[env_ids, 0] = kneel

    def _update_command(self):
        pass


@configclass
class PostureCommandCfg(CommandTermCfg):
    class_type: type = PostureCommand
    resampling_time_range: tuple[float, float] = MISSING
    p_kneel: float = 0.5
    """Probability that a resampled target is the kneel."""
