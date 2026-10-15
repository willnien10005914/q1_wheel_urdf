"""Real RSL-RL CPU compatibility checks; never imports or launches Isaac Sim.

Run this file normally (or with unittest) to test the installed RSL-RL version.
For a 3 -> 5 -> 3 checkpoint round trip, share a directory across three runs::

    <rsl3-python> tests/test_runtime_compat.py --phase create3 --work-dir <directory>
    <rsl5-python> tests/test_runtime_compat.py --phase load5 --work-dir <directory>
    <rsl3-python> tests/test_runtime_compat.py --phase load3 --work-dir <directory>

An isolated package installation can also be selected with PYTHONPATH. The tests
use the installed runners, policy modules, checkpoint readers and normalizers;
only the Isaac config migration hook and unrelated mode asset imports are stubbed.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/reinforcement_learning/rsl_rl")]

import runtime_compat as compat
from rsl_rl.runners import OnPolicyRunner
from skills.policy import FrozenActor


class _Config:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return copy.deepcopy(self.data)


class _Env:
    num_envs = 4
    num_actions = 24
    device = "cpu"
    cfg = {}

    def __init__(self):
        generator = torch.Generator().manual_seed(9001)
        self.obs = TensorDict(
            {"policy": torch.randn(4, 92, generator=generator),
             "critic": torch.randn(4, 111, generator=generator)},
            batch_size=[4],
        )

    def get_observations(self):
        return self.obs


def _load_play_modes():
    """Import the real controller without loading simulator asset definitions."""
    definitions = {
        "unbox.unbox_env_cfg": {"LIE_POSE": {}, "UNBOX_POS_ACTION_SCALE": {}},
        "posture.posture_env_cfg": {"POSTURE_POS_ACTION_SCALE": {}},
        "skate.mdp.observations": {"POSTURE_OBS_INDEX": 85, "GETUP_PHASE_INDEX": 86},
        "skate.skate_env_cfg": {"_position_action_clip": lambda scale: {}},
    }
    stubs = {}
    for suffix, values in definitions.items():
        name = "wheel_humanoid_lab.tasks.manager_based." + suffix
        module = types.ModuleType(name)
        module.__dict__.update(values)
        stubs[name] = module
    path = ROOT / "scripts/reinforcement_learning/rsl_rl/play_modes.py"
    spec = importlib.util.spec_from_file_location("q1_cpu_play_modes", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def _runner(env, normalized, noise):
    modern = compat.rsl_rl_version().major >= 5
    cfg = {
        "class_name": "OnPolicyRunner", "num_steps_per_env": 4, "save_interval": 1,
        "algorithm": {"class_name": "PPO", "rnd_cfg": None, "symmetry_cfg": None},
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
    }
    if modern:
        cfg["actor"] = {
            "class_name": "MLPModel", "hidden_dims": [64, 32], "activation": "elu",
            "obs_normalization": normalized,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": .3, "std_type": noise},
        }
        cfg["critic"] = {
            "class_name": "MLPModel", "hidden_dims": [64, 32], "activation": "elu",
            "obs_normalization": normalized, "distribution_cfg": None,
        }
    else:
        cfg["policy"] = {
            "class_name": "ActorCritic", "actor_hidden_dims": [64, 32], "critic_hidden_dims": [64, 32],
            "activation": "elu", "actor_obs_normalization": normalized,
            "critic_obs_normalization": normalized, "init_noise_std": .3, "noise_std_type": noise,
        }
    original = copy.deepcopy(cfg)
    # Config migration itself requires Isaac Lab and is exercised by simulator
    # validation. This stub only bypasses that unrelated import for real runners.
    lab = types.ModuleType("isaaclab_rl")
    lab.rsl_rl = types.SimpleNamespace()
    with patch.dict(sys.modules, {"isaaclab_rl": lab}):
        migrated = compat.runner_cfg(_Config(cfg))
    assert cfg == original, "runner_cfg modified the caller's config"
    assert migrated["obs_groups"] == {
        "actor" if modern else "policy": ["policy"], "critic": ["critic"]
    }
    torch.manual_seed(42)
    return OnPolicyRunner(env, migrated, log_dir=None, device="cpu")


def _initialize_normalizers(runner):
    generator = torch.Generator().manual_seed(2026)
    actor_obs = torch.randn(128, 92, generator=generator) * 2 + .4
    critic_obs = torch.randn(128, 111, generator=generator) * 3 - .5
    model = compat.policy_module(runner)
    if compat.rsl_rl_version().major >= 5:
        model.obs_normalizer.update(actor_obs)
        runner.alg.critic.obs_normalizer.update(critic_obs)
    else:
        model.actor_obs_normalizer.update(actor_obs)
        model.critic_obs_normalizer.update(critic_obs)


def _save(runner, path):
    if compat.rsl_rl_version().major < 5:
        runner.logger_type = "tensorboard"
    else:
        runner.logger.writer = None
    runner.save(str(path))


def _mode_outputs(modes, env, runner, checkpoint):
    """Exercise actual mode calls with TensorDict and the legacy tensor input."""
    model = compat.policy_module(runner)
    policy = runner.get_inference_policy(device="cpu")
    loaded = modes.LoadedPolicy.wrap(env, None, str(checkpoint), "cpu", runner, policy, model)
    controller = modes.ModeController.__new__(modes.ModeController)
    controller.policies = {name: loaded for name in ("skate", "posture", "slide", "unbox")}
    controller.recovery = None
    controller.pos_joint_names = [f"joint_{i}" for i in range(22)]
    controller.default_pos = torch.zeros(4, 22)
    controller.posture_scale = controller.unbox_scale = torch.ones(22)
    controller.posture_lo = controller.unbox_lo = torch.full((22,), -10.)
    controller.posture_hi = controller.unbox_hi = torch.full((22,), 10.)
    controller.posture_target = 1.
    outputs = {}
    with torch.inference_mode():
        for name in ("skate", "posture", "unbox"):
            for flat in (False, True):
                controller.mode = name
                controller._unbox_t = 0.
                grouped = env.obs.clone()
                obs = grouped["policy"] if flat else grouped
                actual = controller.act(obs, dt=.02)
                expected = policy(grouped)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert tuple(actual.shape) == (4, 24)
                if name == "posture":
                    assert torch.all(grouped["policy"][:, modes.POSTURE_OBS_INDEX] == 1.)
                elif name == "unbox":
                    torch.testing.assert_close(
                        grouped["policy"][:, modes.GETUP_PHASE_INDEX], torch.full((4,), .02 / 6.)
                    )
                outputs[f"{name}_{'tensor' if flat else 'grouped'}"] = actual.clone()
    return outputs


def run_phase(phase, work_dir):
    """Save fixtures or load those saved by the previous RSL-RL version."""
    modern = compat.rsl_rl_version().major >= 5
    expected_major = {"create3": 3, "load5": 5, "load3": 3}.get(phase)
    if expected_major is not None and compat.rsl_rl_version().major != expected_major:
        raise RuntimeError(f"Phase {phase} requires RSL-RL {expected_major}.x; found {compat.rsl_rl_version()}")
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    modes = _load_play_modes()
    results = []
    for normalized in (False, True):
        for noise in ("scalar", "log"):
            label = f"{normalized}_{noise}"
            env = _Env()
            runner = _runner(env, normalized, noise)
            if phase in {"same", "create3"}:
                if normalized:
                    _initialize_normalizers(runner)
                runner.current_learning_iteration = 27
                checkpoint = work_dir / f"{'modern' if modern else 'legacy'}_{label}.pt"
                _save(runner, checkpoint)
                with torch.inference_mode():
                    expected = runner.get_inference_policy(device="cpu")(env.obs).clone()
                expected_modes = _mode_outputs(modes, env, runner, checkpoint)
                torch.save({"base": expected, "modes": expected_modes}, work_dir / f"expected_{label}.pt")
                if phase == "same":
                    # A new runner prevents an accidental no-op load from passing.
                    runner = _runner(env, normalized, noise)
                    compat.load_checkpoint(runner, checkpoint, load_optimizer=True)
            else:
                source_format = "legacy" if modern else "modern"
                checkpoint = work_dir / f"{source_format}_{label}.pt"
                compat.load_checkpoint(runner, checkpoint, load_optimizer=True)
                fixture = torch.load(work_dir / f"expected_{label}.pt", weights_only=True)
                expected, expected_modes = fixture["base"], fixture["modes"]
            assert runner.current_learning_iteration == 27, "checkpoint iteration was lost"
            with torch.inference_mode():
                actual = runner.get_inference_policy(device="cpu")(env.obs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            frozen = FrozenActor(checkpoint, "cpu")
            frozen_actual = frozen(env.obs["policy"])
            torch.testing.assert_close(frozen_actual, expected, rtol=0, atol=0)
            mode_outputs = _mode_outputs(modes, env, runner, checkpoint)
            for name, output in mode_outputs.items():
                torch.testing.assert_close(output, expected_modes[name], rtol=0, atol=0)
            frozen_names = compat.freeze_observation_normalizers(runner)
            assert len(frozen_names) == (2 if normalized else 0)
            model = compat.policy_module(runner)
            roots = (model, runner.alg.critic) if modern else (model,)
            for root in roots:
                for module in root.modules():
                    if hasattr(module, "until") and hasattr(module, "_mean"):
                        original = {key: value.clone() for key, value in module.state_dict().items()}
                        module.train()  # learn() does this after the freeze helper runs.
                        outlier = torch.full((4, module._mean.shape[-1]), 1000.)
                        module.update(outlier)
                        for key, value in module.state_dict().items():
                            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
                        torch.testing.assert_close(module(outlier),
                            (outlier - original["_mean"]) / (original["_std"] + module.eps))
            # Validate the reverse fixture before changing exploration noise.
            if phase == "load5":
                _save(runner, work_dir / f"modern_{label}.pt")
            compat.reset_noise_std(runner, .15)
            model = compat.policy_module(runner)
            owner = model.distribution if modern else model
            parameter = getattr(owner, "std_param" if modern else "std", None)
            if noise == "log":
                parameter = getattr(owner, "log_std_param" if modern else "log_std")
                parameter = parameter.exp()
            torch.testing.assert_close(parameter, torch.full_like(parameter, .15))
            # Discarding the iteration is used by fresh fine-tuning runs.
            compat.load_checkpoint(runner, checkpoint, load_iteration=False)
            assert runner.current_learning_iteration == 0
            results.append({
                "normalization": normalized, "noise": noise,
                "max_error": float((actual - expected).abs().max()),
                "frozen_max_error": float((frozen_actual - expected).abs().max()),
                "mode_cases": len(mode_outputs),
                "frozen_normalizers": len(frozen_names),
            })
    report = {
        "rsl_rl_version": str(compat.rsl_rl_version()), "phase": phase, "cpu_only": True,
        "policy_obs": 92, "critic_obs": 111, "actions": 24, "results": results,
    }
    (work_dir / f"{phase}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class RuntimeCompatTest(unittest.TestCase):
    def test_checkpoint_and_mode_inference(self):
        with tempfile.TemporaryDirectory(prefix="q1-runtime-test-") as directory:
            report = run_phase("same", directory)
        self.assertEqual(len(report["results"]), 4)

    def test_grouped_input_is_preserved(self):
        obs = _Env().obs
        self.assertIs(compat.policy_observations(obs), obs)
        flat = obs["policy"]
        grouped = compat.policy_observations(flat)
        self.assertIs(grouped["policy"], flat)
        self.assertEqual(grouped.batch_size, torch.Size([4]))

    def test_noise_must_be_positive(self):
        for std in (0., -.1):
            with self.assertRaises(ValueError):
                compat.reset_noise_std(None, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("same", "create3", "load5", "load3"), default="same")
    parser.add_argument("--work-dir", type=Path, help="Shared fixture/report directory for all round-trip phases")
    args = parser.parse_args()
    if args.phase != "same" and args.work_dir is None:
        parser.error("--work-dir is required for cross-version phases")
    if args.work_dir is None:
        unittest.main(argv=[sys.argv[0]])
    else:
        print("PASS", json.dumps(run_phase(args.phase, args.work_dir)))
