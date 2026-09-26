"""Mode switching for play.py: one PPO per skill, shared 92-D obs / 24-D action contract.

* ``skate``   : checkpoints/q1_skate_ppo.pt    - two-wheel balance + WASD (unchanged, never retrained)
* ``posture`` : checkpoints/q1_posture_ppo.pt  - kneel on four wheels <-> stand up (posture bit in body[0])
* ``slide``   : checkpoints/q1_slide_ppo.pt    - X2-style push-skating, random wander / circles
* ``unbox``   : checkpoints/q1_unbox_ppo.pt    - box-open supine -> yoga sit-up -> stable kneel

Why separate networks instead of one PPO with a mode input: the skate policy is already stable and its
joint-space action clip (knee +-0.45 rad) cannot reach the kneel (knee ~2.35); folding the kneel into it
would change the action semantics and risk the balance skill. The posture PPO uses a wider leg clip,
which play maps back onto the same joints. The slide PPO is a fine-tune of the skate weights.

Checkpoints are hot-reloaded when the file changes, so a policy that finishes training shows up in the
running Isaac Sim session without a restart.
"""

from __future__ import annotations

import math
import os
import random
import re

import torch
from rsl_rl.runners import OnPolicyRunner

from wheel_humanoid_lab.tasks.manager_based.unbox.unbox_env_cfg import LIE_POSE, UNBOX_POS_ACTION_SCALE
from wheel_humanoid_lab.tasks.manager_based.posture.posture_env_cfg import POSTURE_POS_ACTION_SCALE
from wheel_humanoid_lab.tasks.manager_based.skate.mdp.observations import GETUP_PHASE_INDEX, POSTURE_OBS_INDEX
from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import _position_action_clip


class LoadedPolicy:
    def __init__(self, env, agent_cfg, path: str, device: str):
        self.path = path
        self.device = device
        self._agent_cfg = agent_cfg
        self._env = env
        self.runner: OnPolicyRunner | None = None
        self.policy = None
        self.nn = None
        self.mtime = 0.0
        self.reload()

    @classmethod
    def wrap(cls, env, agent_cfg, path: str, device: str, runner, policy, nn) -> "LoadedPolicy":
        obj = cls.__new__(cls)
        obj.path = path
        obj.device = device
        obj._agent_cfg = agent_cfg
        obj._env = env
        obj.runner = runner
        obj.policy = policy
        obj.nn = nn
        obj.mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
        return obj

    @property
    def available(self) -> bool:
        return self.policy is not None

    def reload(self) -> bool:
        """(Re)load if the checkpoint exists and changed on disk. Returns True when reloaded."""
        if not os.path.isfile(self.path):
            return False
        mtime = os.path.getmtime(self.path)
        if self.policy is not None and mtime <= self.mtime:
            return False
        if self.runner is None:
            self.runner = OnPolicyRunner(self._env, self._agent_cfg.to_dict(), log_dir=None, device=self.device)
        try:
            self.runner.load(self.path)
        except Exception as exc:  # partially written file while training is copying it
            print(f"[WARN] could not load {self.path}: {exc}")
            return False
        self.policy = self.runner.get_inference_policy(device=self.device)
        try:
            self.nn = self.runner.alg.policy
        except AttributeError:
            self.nn = self.runner.alg.actor_critic
        self.mtime = mtime
        return True

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.nn is not None and hasattr(self.nn, "reset"):
            self.nn.reset(mask)


class WanderCommand:
    """Random forward speed + yaw every few seconds (straight runs, arcs, full circles), ramped."""

    def __init__(self, device: str, n_envs: int):
        self.device = device
        self.n = n_envs
        self._cmd = torch.zeros(3, device=device)
        self._target = torch.zeros(3, device=device)
        self._t_left = 0.0

    def reset(self) -> None:
        self._cmd.zero_()
        self._target.zero_()
        self._t_left = 0.0

    def _sample(self) -> None:
        vx = random.uniform(0.5, 1.3)
        kind = random.random()
        if kind < 0.35:
            yaw = 0.0
        elif kind < 0.7:
            yaw = random.choice([-1.0, 1.0]) * random.uniform(0.15, 0.35)
        else:  # circle
            yaw = random.choice([-1.0, 1.0]) * random.uniform(0.4, 0.6)
            vx = random.uniform(0.5, 0.9)
        self._target[0] = vx
        self._target[1] = 0.0
        self._target[2] = yaw
        self._t_left = random.uniform(3.0, 6.0)

    def step(self, dt: float) -> torch.Tensor:
        self._t_left -= dt
        if self._t_left <= 0.0:
            self._sample()
        alpha = 1.0 - math.exp(-dt / 0.6)
        self._cmd += (self._target - self._cmd) * alpha
        return self._cmd.unsqueeze(0).expand(self.n, -1).clone()


class ModeController:
    """Owns the active mode, the posture target, the posture->skate handover and the action remap."""

    STAND_HOLD_STEPS = 25  # 0.5 s at 50 Hz

    def __init__(self, env, policies: dict[str, LoadedPolicy], device: str):
        self.env = env
        self.policies = policies
        self.device = device
        self.mode = "skate"  # skate | posture | slide | lie | unbox
        self.posture_target = 0.0  # 0 stand, 1 kneel — matches training (instant 0/1, never ramped)
        self._stand_hold = 0
        self._unbox_t = 0.0
        self.KNEEL_HOLD_STEPS = 25  # 0.5 s at 50 Hz, then hand off to posture PPO
        u = env.unwrapped
        self.robot = u.scene["robot"]
        term = u.action_manager.get_term("joint_pos")
        self.pos_joint_names = list(term._joint_names)
        ids = [self.robot.joint_names.index(n) for n in self.pos_joint_names]
        self._pos_ids = torch.tensor(ids, device=device)
        self.default_pos = self.robot.data.default_joint_pos[:, self._pos_ids].clone()
        scale = torch.zeros(len(ids), device=device)
        lo = torch.zeros(len(ids), device=device)
        hi = torch.zeros(len(ids), device=device)
        clip = _position_action_clip(POSTURE_POS_ACTION_SCALE)
        for i, n in enumerate(self.pos_joint_names):
            scale[i] = next(s for pat, s in POSTURE_POS_ACTION_SCALE.items() if re.fullmatch(pat, n))
            lo[i], hi[i] = clip[n]
        self.posture_scale = scale
        self.posture_lo = lo
        self.posture_hi = hi
        gscale = torch.zeros(len(ids), device=device)
        glo = torch.zeros(len(ids), device=device)
        ghi = torch.zeros(len(ids), device=device)
        gclip = _position_action_clip(UNBOX_POS_ACTION_SCALE)
        for i, n in enumerate(self.pos_joint_names):
            gscale[i] = next(s for pat, s in UNBOX_POS_ACTION_SCALE.items() if re.fullmatch(pat, n))
            glo[i], ghi[i] = gclip[n]
        self.unbox_scale = gscale
        self.unbox_lo = glo
        self.unbox_hi = ghi
        lie = self.default_pos.clone()
        for i, n in enumerate(self.pos_joint_names):
            for pat, val in LIE_POSE.items():
                if re.fullmatch(pat, n):
                    lie[:, i] = float(val)
        self.lie_targets = lie
        self.processed_override: torch.Tensor | None = None  # (n_envs, 22) targets for the joint_pos term
        for n in ("l_knee_joint", "l_hip_pitch_joint"):
            assert n in self.pos_joint_names
        self._knee = torch.tensor([self.pos_joint_names.index("l_knee_joint"), self.pos_joint_names.index("r_knee_joint")], device=device)
        self._hip = torch.tensor([self.pos_joint_names.index("l_hip_pitch_joint"), self.pos_joint_names.index("r_hip_pitch_joint")], device=device)
        self.wander = WanderCommand(device, u.num_envs)

    # ---------------------------------------------------------------- mode requests
    @property
    def has_posture_policy(self) -> bool:
        return self.policies["posture"].available

    @property
    def has_slide_policy(self) -> bool:
        return self.policies["slide"].available

    @property
    def has_unbox_policy(self) -> bool:
        p = self.policies.get("unbox") or self.policies.get("getup")
        return p is not None and p.available

    def request(self, name: str) -> str | None:
        """kneel | stand | slide | skate | lie | unbox/getup -> message or None when nothing changed."""
        if name == "kneel":
            if not self.has_posture_policy:
                return None
            if self.mode == "posture" and self.posture_target == 1.0:
                return None
            self._enter("posture")
            self.posture_target = 1.0
            return "Kneel: posture PPO (same command it was trained on)."
        if name == "stand":
            if not self.has_posture_policy:
                return None
            if self.mode != "posture":
                return None
            self.posture_target = 0.0
            self._stand_hold = 0
            return "Stand up: posture PPO -> skate pose, then skate PPO."
        if name == "slide":
            if self.mode == "slide":
                return None
            if self.mode == "posture":
                return None  # stand up first
            self._enter("slide")
            self.wander.reset()
            msg = "Slide mode: push-skate PPO, random wander (WASD overrides)."
            if not self.has_slide_policy:
                msg = "Slide mode: q1_slide_ppo.pt not trained yet, wandering with the skate PPO."
            return msg
        if name == "lie":
            self._enter("lie")
            self._unbox_t = 0.0
            return "Lie face up (box-open). Press Unbox to sit up onto the knee rollers."
        if name in {"getup", "unbox"}:
            if not self.has_unbox_policy:
                return "Unbox: q1_unbox_ppo.pt is not trained yet."
            self._enter("unbox")
            self._unbox_t = 0.0
            return "Unbox: tuck, yoga sit-up, knee rollers, wheel scoot, waist to a stable kneel."
        if name == "skate":
            if self.mode == "skate":
                return None
            if self.mode == "posture" and self.posture_target == 1.0:
                return None
            self._enter("skate")
            return "Skate mode: balance PPO + WASD."
        return None

    def _enter(self, mode: str) -> None:
        self.mode = mode
        self._stand_hold = 0
        self.processed_override = None
        for p in self.policies.values():
            p.reset()

    def reset(self) -> None:
        self._enter("skate")
        self.posture_target = 0.0
        self.wander.reset()

    # ---------------------------------------------------------------- per-step
    @property
    def blocking(self) -> bool:
        """WASD is ignored while squatting down or standing up. A settled kneel accepts it."""
        if self.mode in {"lie", "unbox", "getup"}:
            return True
        if self.mode != "posture":
            return False
        if self.posture_target != 1.0:
            return True
        return float(self.robot.data.root_pos_w[0, 2]) > 0.55

    @property
    def suppress_terminations(self) -> bool:
        return self.mode in {"posture", "lie", "unbox", "getup"}

    @property
    def label(self) -> str:
        if self.mode in {"lie", "unbox", "getup"}:
            return self.mode
        if self.mode == "posture":
            z = float(self.robot.data.root_pos_w[0, 2])
            if self.posture_target == 1.0:
                return "kneel" if z < 0.55 else "kneeling"
            return "standing"
        return self.mode

    def active_policy(self) -> LoadedPolicy:
        if self.mode in {"unbox", "getup"} and self.has_unbox_policy:
            return self.policies.get("unbox") or self.policies["getup"]
        if self.mode == "posture":
            return self.policies["posture"]
        if self.mode == "slide" and self.has_slide_policy:
            return self.policies["slide"]
        return self.policies["skate"]

    def poll_reload(self) -> None:
        for name, p in self.policies.items():
            if p.reload():
                print(f"[INFO] Reloaded {name} policy from {p.path}")

    def act(self, obs, dt: float = 0.02) -> torch.Tensor:
        pol = self.active_policy()
        if self.mode == "lie":
            self.processed_override = self.lie_targets
            pol = self.policies["skate"]
            return torch.zeros(self.env.unwrapped.num_envs, 24, device=self.device)
        if self.mode in {"unbox", "getup"}:
            self._unbox_t += dt
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            policy_obs[:, GETUP_PHASE_INDEX] = min(self._unbox_t / 6.0, 1.0)
            actions = pol.policy(obs)
            a_pos = actions[:, : len(self.pos_joint_names)]
            targets = self.default_pos + self.unbox_scale * a_pos
            self.processed_override = torch.maximum(torch.minimum(targets, self.unbox_hi), self.unbox_lo)
            return actions
        if self.mode == "posture":
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            policy_obs[:, POSTURE_OBS_INDEX] = self.posture_target
            actions = pol.policy(obs)
            a_pos = actions[:, : len(self.pos_joint_names)]
            targets = self.default_pos + self.posture_scale * a_pos
            self.processed_override = torch.maximum(torch.minimum(targets, self.posture_hi), self.posture_lo)
            return actions
        self.processed_override = None
        return pol.policy(obs)

    def after_step(self, dones: torch.Tensor) -> str | None:
        """Posture -> skate handover once the robot is back in the skate pose and still."""
        pol = self.active_policy()
        if self.mode in {"unbox", "getup"}:
            z = float(self.robot.data.root_pos_w[0, 2])
            g = self.robot.data.projected_gravity_b[0]
            gx = float(g[0])
            tilt = math.acos(max(-1.0, min(1.0, -float(g[2]))))
            if 0.36 < z < 0.60 and tilt < 0.50 and gx < 0.40:
                self._stand_hold += 1
            else:
                self._stand_hold = 0
            if self._stand_hold >= self.KNEEL_HOLD_STEPS:
                if self.has_posture_policy:
                    self._enter("posture")
                    self.posture_target = 1.0
                    return "Unbox reached a stable kneel — posture PPO holding. Press Stand up."
                return "Unbox reached a kneel (no posture PPO to stand)."
            return None
        if self.mode != "posture":
            pol.reset(dones)
            return None
        if self.posture_target != 0.0:
            return None
        q = self.robot.data.joint_pos[:, self._pos_ids]
        dq_knee = (q[0, self._knee] - self.default_pos[0, self._knee]).abs().max().item()
        dq_hip = (q[0, self._hip] - self.default_pos[0, self._hip]).abs().max().item()
        z = float(self.robot.data.root_pos_w[0, 2])
        g = self.robot.data.projected_gravity_b[0]
        tilt = math.acos(max(-1.0, min(1.0, -float(g[2]))))
        speed = float(self.robot.data.root_lin_vel_b[0, :2].norm())
        if z > 0.78 and tilt < 0.15 and dq_knee < 0.2 and dq_hip < 0.2 and speed < 0.25:
            self._stand_hold += 1
        else:
            self._stand_hold = 0
        if self._stand_hold >= self.STAND_HOLD_STEPS:
            self._enter("skate")
            return "Standing skate pose reached - skate PPO balancing."
        return None
