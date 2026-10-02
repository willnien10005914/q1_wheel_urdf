"""Run contact-gated recovery-v3 PPO inside the skate play env.

The recovery actor expects 118-D observations (skate proprio + recovery command block)
and 24-D residuals that ContactPosition / ContactWheel actions turn into motor targets.
Skate play keeps its own action terms, so this bridge rebuilds those targets and writes
them through ModeController.processed_override (+ wheel override).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from tensordict import TensorDict

try:
    from rsl_rl.modules import ActorCritic  # rsl-rl < 5
except ImportError:  # pragma: no cover - rsl-rl >= 5 removed ActorCritic
    ActorCritic = None  # type: ignore

from wheel_humanoid_lab.tasks.manager_based.recovery_v3 import mdp as recovery_mdp


SKATE_COMMAND_DIM = 14
RECOVERY_ACTOR_OBS = 118
RECOVERY_CRITIC_OBS = 128
RECOVERY_ACTIONS = 24


def _default_checkpoint(project_root: str, mode: int | None = None) -> str:
    """Prefer mode-specific split PPO, then mixed v3, then the last full run."""
    names = []
    if mode == 0:
        names.append("q1_recovery_v3_supine_ppo.pt")
    elif mode == 1:
        names.append("q1_recovery_v3_prone_ppo.pt")
    else:
        names.extend(["q1_recovery_v3_supine_ppo.pt", "q1_recovery_v3_prone_ppo.pt"])
    names.append("q1_recovery_contact_v3_ppo.pt")
    for name in names:
        path = os.path.join(project_root, "checkpoints", name)
        if os.path.isfile(path):
            return path
    return os.path.join(
        project_root,
        "logs/rsl_rl/q1_recovery_contact_v3/2026-09-29_20-43-53_v3_smooth_arms_full_4096/model_19999.pt",
    )


def load_recovery_actor(checkpoint: str, device: str):
    if ActorCritic is None:
        raise RuntimeError("rsl-rl >= 5 removed ActorCritic; use evaluate_recovery_v3 / converted MLPModel load instead")

    obs = TensorDict(
        {
            "policy": torch.zeros(1, RECOVERY_ACTOR_OBS, device=device),
            "critic": torch.zeros(1, RECOVERY_CRITIC_OBS, device=device),
        },
        batch_size=[1],
        device=device,
    )
    ac = ActorCritic(
        obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=RECOVERY_ACTIONS,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        init_noise_std=0.05,
        noise_std_type="scalar",
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    ac.load_state_dict(state)
    ac.to(device)
    ac.eval()
    return ac


def _policy_tensor(obs: Any) -> torch.Tensor:
    """RslRlVecEnvWrapper.get_observations() returns a TensorDict, not a plain tensor.

    TensorDict.shape is the *batch* shape (e.g. (1,)), so callers must index ["policy"].
    """
    if torch.is_tensor(obs):
        return obs
    try:
        policy = obs["policy"]
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"expected TensorDict/dict with 'policy' or a Tensor, got {type(obs)}") from exc
    if not torch.is_tensor(policy):
        raise TypeError(f"obs['policy'] is {type(policy)}, expected Tensor")
    return policy


class RecoveryV3PlayBridge:
    """Lie supine/prone, then run recovery PPO until stand, then hand back to skate/slide."""

    STAND_HOLD_S = 0.8

    def __init__(self, env, checkpoint: str | None, device: str):
        self.env = env.unwrapped
        self.device = device
        self.checkpoint = checkpoint
        self.policy: ActorCritic | None = None
        self.mixed_policy: ActorCritic | None = None
        self.mixed_checkpoint: str | None = None
        self.policies: dict[int, ActorCritic] = {}
        self.checkpoints: dict[int, str] = {}
        self.available = False
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        for mode, name in ((0, "q1_recovery_v3_supine_ppo.pt"), (1, "q1_recovery_v3_prone_ppo.pt")):
            path = os.path.join(project_root, "checkpoints", name)
            if os.path.isfile(path):
                try:
                    self.policies[mode] = load_recovery_actor(path, device)
                    self.checkpoints[mode] = path
                    print(f"[INFO] Loaded recovery split PPO mode={mode}: {path}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] recovery split load failed ({path}): {exc}")
        mixed = os.path.join(project_root, "checkpoints", "q1_recovery_contact_v3_ppo.pt")
        # Prefer explicit mixed ckpt; avoid treating a split file passed as RECOVERY_CHECKPOINT
        # as a universal fallback (would run the wrong orientation's policy).
        mixed_candidate = None
        if checkpoint and os.path.isfile(checkpoint):
            base = os.path.basename(checkpoint)
            if "supine" not in base and "prone" not in base:
                mixed_candidate = checkpoint
        if mixed_candidate is None and os.path.isfile(mixed):
            mixed_candidate = mixed
        if mixed_candidate:
            try:
                self.mixed_policy = load_recovery_actor(mixed_candidate, device)
                self.mixed_checkpoint = mixed_candidate
                print(f"[INFO] Loaded recovery mixed PPO fallback: {mixed_candidate}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] recovery mixed load failed ({mixed_candidate}): {exc}")
        self.available = bool(self.policies) or self.mixed_policy is not None
        self.policy = self.mixed_policy or (next(iter(self.policies.values())) if self.policies else None)
        self.checkpoint = self.mixed_checkpoint or (
            next(iter(self.checkpoints.values())) if self.checkpoints else checkpoint
        )
        self.lie_mode = 0  # 0=supine, 1=prone
        self.running = False
        self.lying = False
        self.processed_override: torch.Tensor | None = None
        self.processed_wheel: torch.Tensor | None = None
        self._lie_hold_q: torch.Tensor | None = None  # frozen floor pose while waiting for Recovery PPO
        self._pos_ids: torch.Tensor | None = None
        self._wheel_ids: torch.Tensor | None = None
        self._arm_action_ids: list[int] | None = None
        self._hip_roll_action_ids: list[int] | None = None

    def _state(self):
        return recovery_mdp.state(self.env)

    def _ensure_ids(self) -> None:
        if self._pos_ids is not None:
            return
        s = self._state()
        self._pos_ids = torch.as_tensor(s.pos_ids, device=self.device, dtype=torch.long)
        wheel = self.env.scene["robot"].find_joints(["l_wheel_joint", "r_wheel_joint"], preserve_order=True)[0]
        self._wheel_ids = torch.as_tensor(wheel, device=self.device, dtype=torch.long)
        names = [self.env.scene["robot"].joint_names[int(i)] for i in self._pos_ids.tolist()]
        self._arm_action_ids = [i for i, n in enumerate(names) if ("shoulder" in n or "elbow" in n)]
        self._hip_roll_action_ids = [i for i, n in enumerate(names) if "hip_roll" in n]

    def _active_policy(self) -> ActorCritic | None:
        """Use orientation-specific split PPO; only fall back to mixed, never the other mode."""
        if self.lie_mode in self.policies:
            return self.policies[self.lie_mode]
        return self.mixed_policy

    def place_lie(self, mode: int) -> str:
        """Teleport into supine (0) or prone (1) and hold the floor pose (no PPO yet)."""
        mode = 0 if int(mode) <= 0 else 1
        self.lie_mode = mode
        self.running = False
        self.lying = True
        self.processed_wheel = None
        ids = torch.arange(self.env.num_envs, device=self.device)
        recovery_mdp.reset(self.env, ids, mode=mode)
        s = self._state()
        s.cap = 4
        s.stand_allowed[:] = s.stand_enabled
        self._ensure_ids()
        # Freeze the trajectory frame-0 pose. Calling s.update() while waiting would
        # blend motor priors toward stage-0 arm-plant (shoulder_pitch→~2.1) and raise arms.
        self._lie_hold_q = s.command[:, self._pos_ids].clone()
        self.processed_override = self._lie_hold_q.clone()
        self.processed_wheel = torch.zeros(self.env.num_envs, 2, device=self.device)
        label = "supine" if mode == 0 else "prone"
        ready = "ready" if self._active_policy() is not None else "MISSING ckpt"
        return f"Lie {label} ({ready}): hold floor pose. Press Recovery PPO to get up."

    def start_recovery(self) -> str:
        policy = self._active_policy()
        label = "supine" if self.lie_mode == 0 else "prone"
        if policy is None:
            need = "q1_recovery_v3_supine_ppo.pt" if self.lie_mode == 0 else "q1_recovery_v3_prone_ppo.pt"
            return f"Recovery PPO ({label}): checkpoint missing ({need} or mixed q1_recovery_contact_v3_ppo.pt)."
        if not self.lying:
            # Default to last selected orientation (or supine).
            self.place_lie(self.lie_mode)
        self.policy = policy
        if self.lie_mode in self.checkpoints:
            self.checkpoint = self.checkpoints[self.lie_mode]
        elif self.mixed_checkpoint:
            self.checkpoint = self.mixed_checkpoint
        ids = torch.arange(self.env.num_envs, device=self.device)
        # Re-seat so stage timers and contacts start clean from the chosen orientation.
        recovery_mdp.reset(self.env, ids, mode=self.lie_mode)
        s = self._state()
        s.cap = 4
        s.stand_allowed[:] = s.stand_enabled
        self._ensure_ids()
        self.running = True
        self.lying = True
        self._lie_hold_q = None
        self.processed_override = s.command[:, self._pos_ids].clone()
        self.processed_wheel = torch.zeros(self.env.num_envs, 2, device=self.device)
        ckpt = os.path.basename(self.checkpoint) if self.checkpoint else "?"
        return f"Recovery PPO ({label} / {ckpt}): contact-gated getup → stand."

    def stop(self) -> None:
        self.running = False
        self.lying = False
        self._lie_hold_q = None
        self.processed_override = None
        self.processed_wheel = None

    def _process_position(self, actions: torch.Tensor) -> torch.Tensor:
        """Mirror ContactPositionAction.process_actions without owning the action term."""
        s = self._state()
        robot = self.env.scene["robot"]
        lim = robot.data.soft_joint_pos_limits[:, self._pos_ids]
        mid = lim.mean(-1)
        half = (lim[:, :, 1] - lim[:, :, 0]) / 2
        bias = torch.atanh(((s.command[:, self._pos_ids] - mid) / half).clamp(-0.98, 0.98))
        calm = (s.stage >= 3) & (s.arm_force.max(-1).values < 30)
        gain = torch.where(calm, 0.2, 0.65)[:, None]
        a_pos = actions[:, : self._pos_ids.numel()]
        processed = mid + half * torch.tanh(gain * a_pos + bias)
        if self._arm_action_ids:
            idx = torch.tensor(self._arm_action_ids, device=self.device, dtype=torch.long)
            prior = s.command[:, self._pos_ids][:, idx]
            blend = torch.where(calm, 0.85, 0.0)[:, None]
            arm = processed[:, idx]
            processed = processed.clone()
            processed[:, idx] = blend * prior + (1.0 - blend) * arm
        if getattr(self, "_hip_roll_action_ids", None):
            idx = torch.tensor(self._hip_roll_action_ids, device=self.device, dtype=torch.long)
            prior = s.command[:, self._pos_ids][:, idx]
            processed = processed.clone()
            processed[:, idx] = 0.75 * prior + 0.25 * processed[:, idx]
        return processed

    def _process_wheel(self, actions: torch.Tensor) -> torch.Tensor:
        s = self._state()
        residual = (8.0 * actions[:, -2:]).clamp(-8.0, 8.0)
        balance = recovery_mdp.balance_velocity(s).clamp(-25.0, 25.0)
        return (residual + balance[:, None]).clamp(-25.0, 25.0)

    def _recovery_obs(self, skate_policy_obs: torch.Tensor) -> torch.Tensor:
        cmd = recovery_mdp.observation(self.env)
        if skate_policy_obs.ndim != 2 or skate_policy_obs.shape[-1] < SKATE_COMMAND_DIM:
            raise RuntimeError(f"skate policy obs too short: {tuple(skate_policy_obs.shape)}")
        if cmd.shape[0] != skate_policy_obs.shape[0]:
            raise RuntimeError(f"recovery cmd batch {tuple(cmd.shape)} != policy {tuple(skate_policy_obs.shape)}")
        return torch.cat((skate_policy_obs[:, :-SKATE_COMMAND_DIM], cmd), dim=-1)

    def act(self, skate_obs: Any) -> torch.Tensor:
        """Return raw 24-D actions for env.step; targets live in processed_* overrides."""
        n = self.env.num_envs
        zeros = torch.zeros(n, RECOVERY_ACTIONS, device=self.device)
        self._ensure_ids()
        s = self._state()
        if self.lying and not self.running:
            # Hold frozen floor pose only — do not advance recovery stage/motor priors.
            hold = self._lie_hold_q
            if hold is None:
                hold = self._state().command[:, self._pos_ids].clone()
                self._lie_hold_q = hold
            self.processed_override = hold.clone()
            self.processed_wheel = torch.zeros(n, 2, device=self.device)
            return zeros
        if not self.running or self._active_policy() is None:
            self.processed_override = None
            self.processed_wheel = None
            return zeros

        policy = self._active_policy()
        policy_obs = _policy_tensor(skate_obs)
        rec_obs = self._recovery_obs(policy_obs)
        td = TensorDict({"policy": rec_obs}, batch_size=[n], device=self.device)
        with torch.inference_mode():
            actions = policy.act_inference(td)
        self.processed_override = self._process_position(actions)
        self.processed_wheel = self._process_wheel(actions)
        return actions

    def after_step(self) -> str | None:
        if not self.running:
            return None
        s = self._state()
        s.update()
        if bool((s.stand_hold >= self.STAND_HOLD_S).any().item()) or bool(s.stood.any().item()):
            self.running = False
            self.lying = False
            self.processed_override = None
            self.processed_wheel = None
            return "Recovery reached stand — skate/slide WASD unlocked."
        return None

    @property
    def blocking(self) -> bool:
        return self.lying or self.running

    @property
    def label(self) -> str:
        if self.running:
            return "recovery"
        if self.lying:
            return "supine" if self.lie_mode == 0 else "prone"
        return "idle"
