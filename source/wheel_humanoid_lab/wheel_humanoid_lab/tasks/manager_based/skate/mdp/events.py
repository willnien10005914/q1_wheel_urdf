"""Domain randomization events with restore-then-sample semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_posture_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    kneel_pose: dict[str, float],
    kneel_root_z: float,
    p_kneel: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Start a random subset of episodes already in the four-wheel kneel (the rest keep the stand
    default from ``reset_robot_joints``). Must run after ``reset_base`` / ``reset_robot_joints``."""
    import re

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None or len(env_ids) == 0:
        return
    pick = env_ids[torch.rand(len(env_ids), device=env.device) < p_kneel]
    if len(pick) == 0:
        return
    q = asset.data.default_joint_pos[pick].clone()
    for i, name in enumerate(asset.joint_names):
        for pat, val in kneel_pose.items():
            if re.fullmatch(pat, name):
                q[:, i] = float(val)
    qd = torch.zeros_like(q)
    asset.write_joint_state_to_sim(q, qd, env_ids=pick)
    root = asset.data.root_state_w[pick].clone()
    root[:, 2] = kneel_root_z + env.scene.env_origins[pick, 2]
    root[:, 7:] = 0.0
    asset.write_root_pose_to_sim(root[:, :7], env_ids=pick)
    asset.write_root_velocity_to_sim(root[:, 7:], env_ids=pick)


class randomize_com(ManagerTermBase):
    """CoM offset DR that never accumulates.

    Isaac Lab's ``randomize_rigid_body_com`` does ``coms += sample`` on the *current* CoM, so in reset
    mode the offset random-walks over episodes (the Microduck CoM bug). Here the USD CoM is captured
    once and every call writes ``default + sample``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]
        self.default_coms = self.asset.root_physx_view.get_coms().clone()  # (N, B, 7) on CPU
        if asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(self.asset.num_bodies, dtype=torch.long)
        else:
            self.body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long)

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, com_range: dict[str, tuple[float, float]]):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs)
        env_ids = env_ids.cpu()
        ranges = torch.tensor([com_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")])
        if float(ranges.abs().max()) == 0.0 and not hasattr(self, "_dirty"):
            return
        self._dirty = True
        coms = self.asset.root_physx_view.get_coms().clone()
        coms[env_ids[:, None], self.body_ids[None, :], :3] = self.default_coms[env_ids[:, None], self.body_ids[None, :], :3]
        sample = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device="cpu")
        coms[env_ids[:, None], self.body_ids[None, :], :3] += sample
        self.asset.root_physx_view.set_coms(coms, env_ids)
