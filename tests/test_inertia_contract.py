#!/usr/bin/env python
"""Bounded native Q1 ManagerBasedRLEnv source-to-PhysX contract.

    ./run_isaac.sh tests/test_inertia_contract.py --headless --num_envs 2

Use --source-urdf /path/to/BU10_robot_Assets/urdf/quanta_robot_wheeled.urdf
to additionally compare the relocated URDF with the original CAD export. The
default uses the checked-in, mesh-relocated URDF. No checkpoint is required.
"""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--source-urdf", type=Path)
parser.add_argument("--report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.steps <= 64:
    parser.error("--steps must be between 1 and 64")
if args.num_envs < 1:
    parser.error("--num_envs must be positive")
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from pxr import UsdPhysics  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import wheel_humanoid_lab.tasks  # noqa: F401, E402
from wheel_humanoid_lab.assets.q1_inertia import (  # noqa: E402
    Q1_INERTIA_ACTION_DIM, Q1_INERTIA_ACTOR_OBS_DIM, Q1_INERTIA_CRITIC_OBS_DIM,
    Q1_INERTIA_JOINT_ORDER, Q1_INERTIA_POSITION_JOINTS, Q1_INERTIA_WHEEL_JOINTS,
    Q1_INERTIA_MIMIC_JOINTS, Q1_INERTIA_MIMIC_DRIVERS, Q1_INERTIA_MIMIC_MULTIPLIERS,
    Q1_INERTIA_URDF,
)

TASK = "Isaac-Q1-Inertia-Skate-v0"
EXPECTED_MASS = 41.052262573
failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        failures.append(message)


def rotation_wxyz(quaternion):
    """Column-vector rotation; compare full tensors, never ambiguous eigenvector signs."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def source_contract(path):
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        origin = inertial.find("origin")
        com = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                             [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                             [-sp, cp*sr, cp*cr]])
        a = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
        tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                           [a["ixy"], a["iyy"], a["iyz"]],
                           [a["ixz"], a["iyz"], a["izz"]]])
        links[link.get("name")] = (float(inertial.find("mass").get("value")), com,
                                    rotation @ tensor @ rotation.T)
    mimics = {}
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics[joint.get("name")] = (mimic.get("joint"), float(mimic.get("multiplier", 1)),
                                         float(mimic.get("offset", 0)))
    return links, mimics


def main():
    source, mimics = source_contract(args.source_urdf or Q1_INERTIA_URDF)
    integrated, integrated_mimics = source_contract(Q1_INERTIA_URDF)
    check(set(source) == set(integrated), "all 28 source links preserved")
    for name in source.keys() & integrated.keys():
        check(all(np.array_equal(a, b) for a, b in zip(source[name], integrated[name])),
              f"{name}: source mass, COM and inertia preserved by mesh relocation")
    check(mimics == integrated_mimics, "all source mimic declarations preserved")
    check(len(source) == 28 and len(mimics) == 4, "28 links and four source mimics")
    check(abs(sum(item[0] for item in source.values()) - EXPECTED_MASS) < 1e-9,
          f"source mass = {EXPECTED_MASS:.9f} kg")
    expected_mimics = {name: (driver, mult, 0.0) for name, driver, mult in zip(
        Q1_INERTIA_MIMIC_JOINTS, Q1_INERTIA_MIMIC_DRIVERS, Q1_INERTIA_MIMIC_MULTIPLIERS)}
    check(mimics == expected_mimics, "all four mimic drivers, multipliers and zero offsets match source")

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = 4107
    cfg.observations.policy.enable_corruption = False
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        check(isinstance(env, ManagerBasedRLEnv), "native task uses ManagerBasedRLEnv")
        robot = env.scene["robot"]
        obs, _ = env.reset(seed=4107)
        check(robot.num_joints == 27, f"27 movable joints, got {robot.num_joints}")
        check(set(robot.joint_names) == set(Q1_INERTIA_JOINT_ORDER + Q1_INERTIA_MIMIC_JOINTS),
              "23 policy joints and all four followers resolved by name")
        check(env.action_manager.total_action_dim == Q1_INERTIA_ACTION_DIM == 23, "23 action channels")
        check(obs["policy"].shape == (args.num_envs, Q1_INERTIA_ACTOR_OBS_DIM)
              and Q1_INERTIA_ACTOR_OBS_DIM == 78, "actor observation shape = N x 78")
        check(obs["critic"].shape == (args.num_envs, Q1_INERTIA_CRITIC_OBS_DIM)
              and Q1_INERTIA_CRITIC_OBS_DIM == 81, "critic observation shape = N x 81")
        check(env.action_manager.get_term("joint_pos")._joint_names == Q1_INERTIA_POSITION_JOINTS,
              "21 position action channels in declared order")
        check(env.action_manager.get_term("wheel_vel")._joint_names == Q1_INERTIA_WHEEL_JOINTS,
              "two wheel velocity channels in declared order")

        # get_inertias() returns COM tensors expressed in each rigid-body link frame.
        # USD stores eigenvalues and a principal-axes quaternion: reconstruct R D R^T.
        masses = robot.root_physx_view.get_masses().cpu().numpy()
        tensors = robot.root_physx_view.get_inertias().cpu().numpy().reshape(args.num_envs, -1, 3, 3)
        coms = robot.root_physx_view.get_coms().cpu().numpy()
        stage = sim_utils.get_current_stage()
        max_tensor_error = 0.0
        check(set(robot.body_names) == set(source), "PhysX contains all source links without merging")
        for body_id, name in enumerate(robot.body_names):
            mass, com, tensor = source[name]
            api = UsdPhysics.MassAPI(stage.GetPrimAtPath(robot.root_physx_view.link_paths[0][body_id]))
            usd_com = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=np.float64)
            principal = api.GetPrincipalAxesAttr().Get()
            rotation = rotation_wxyz((principal.GetReal(), *principal.GetImaginary()))
            usd_tensor = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
            max_tensor_error = max(max_tensor_error, float(np.max(np.abs(usd_tensor - tensor))),
                                   float(np.max(np.abs(tensors[:, body_id] - tensor))))
            check(np.isclose(api.GetMassAttr().Get(), mass, rtol=2e-6, atol=1e-8)
                  and np.allclose(masses[:, body_id], mass, rtol=2e-6, atol=1e-8), f"{name}: USD/PhysX mass matches source")
            check(np.allclose(usd_com, com, rtol=2e-6, atol=2e-8)
                  and np.allclose(coms[:, body_id, :3], com, rtol=2e-6, atol=2e-8), f"{name}: USD/PhysX COM matches source")
            check(np.allclose(usd_tensor, tensor, rtol=2e-5, atol=2e-8)
                  and np.allclose(tensors[:, body_id], tensor, rtol=2e-5, atol=2e-8),
                  f"{name}: reconstructed USD/PhysX inertia tensor matches source")
        check(np.allclose(masses.sum(axis=1), EXPECTED_MASS, rtol=0, atol=2e-5),
              f"PhysX mass = {masses[0].sum():.9f} kg in every environment")

        mimic = env.action_manager.get_term("mimic")
        follower_ids, follower_names = robot.find_joints(Q1_INERTIA_MIMIC_JOINTS, preserve_order=True)
        driver_ids, driver_names = robot.find_joints(Q1_INERTIA_MIMIC_DRIVERS, preserve_order=True)
        multipliers = torch.tensor(Q1_INERTIA_MIMIC_MULTIPLIERS, device=env.device).unsqueeze(0)
        check(follower_names == Q1_INERTIA_MIMIC_JOINTS and driver_names == Q1_INERTIA_MIMIC_DRIVERS,
              "all four follower and driver channels resolved in source order")
        check(mimic.action_dim == 0, "mimic servos consume zero policy channels")
        finite = True
        targets_exact = True
        max_tracking_error = 0.0
        substep_calls = 0
        apply_mimic = mimic.apply_actions

        def checked_apply_mimic():
            nonlocal targets_exact, substep_calls
            expected_q = robot.data.joint_pos[:, driver_ids] * multipliers
            expected_dq = robot.data.joint_vel[:, driver_ids] * multipliers
            apply_mimic()
            targets_exact &= bool(torch.equal(mimic.position_target, expected_q)
                                  and torch.equal(robot.data.joint_pos_target[:, follower_ids], expected_q)
                                  and torch.equal(robot.data.joint_vel_target[:, follower_ids], expected_dq))
            substep_calls += 1

        # Observe the actual manager calls before PhysX, rather than reapplying the
        # relation after a step (which would miss broken substep scheduling).
        mimic.apply_actions = checked_apply_mimic
        with torch.inference_mode():
            for step in range(args.steps):
                # Small deterministic excitation exercises every policy channel and both wheels.
                phase = torch.arange(23, device=env.device) * 0.37 + step * 0.13
                action = (0.10 * torch.sin(phase)).expand(args.num_envs, -1).clone()
                obs, reward, terminated, truncated, _ = env.step(action)
                expected_q = robot.data.joint_pos[:, driver_ids] * multipliers
                tracking_error = robot.data.joint_pos[:, follower_ids] - expected_q
                finite &= bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
                               and torch.isfinite(reward).all() and torch.isfinite(tracking_error).all()
                               and torch.isfinite(robot.data.joint_vel).all()
                               and torch.isfinite(robot.data.root_state_w).all()
                               and not env.termination_manager.get_term("nonfinite").any())
                max_tracking_error = max(max_tracking_error, float(tracking_error.abs().max()))
                if not finite:
                    break
        check(substep_calls == args.steps * cfg.decimation,
              f"mimic servos updated on every physics substep ({substep_calls})")
        check(targets_exact, "all four mimic position/velocity targets exactly follow source relations")
        check(finite, f"finite observations, rewards, rigid state and all four mimic tracking errors for {args.steps} steps")
        print(f"Maximum finite-gain mimic tracking error: {max_tracking_error:.6f} rad", flush=True)
        report = {"task": TASK, "num_envs": args.num_envs, "steps": args.steps, "source_urdf": str(args.source_urdf or Q1_INERTIA_URDF),
                  "mass_kg": masses[0].sum().item(), "action_dim": 23, "actor_obs_dim": 78, "critic_obs_dim": 81,
                  "joint_count": robot.num_joints, "source_inertia_max_abs_error": max_tensor_error,
                  "mimic_targets_exact": targets_exact, "mimic_max_tracking_error_rad": max_tracking_error,
                  "mimic_physics_substep_calls": substep_calls,
                  "finite": finite, "failures": failures, "pass": not failures}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()
    return int(bool(failures))


if __name__ == "__main__":
    try:
        result = main()
    finally:
        app.close()
    sys.exit(result)
