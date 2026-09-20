#!/usr/bin/env python
"""Obs/action contract + actuator-group unit test for Isaac-Q1-Skate-v0 (runs inside Isaac Sim).

    ./run_isaac.sh tests/test_skate_contract.py --num_envs 4

Checks (all name based, none of them index PhysX joints by position):
  * the articulation has the 24 contract joints + 2 passive knee rollers, all resolved by name,
  * every contract joint is driven by a CubeMarsActuator grouped by SKU, wheels in velocity mode,
  * PhysX joint friction is zero for the CubeMars joints (friction lives in the actuator),
  * policy obs = 78 proprio + 14 command = 92, critic obs is wider, action = 24 = 22 pos + 2 wheel vel,
  * command block slots: twist(3) | head(4)=0 | body(6)=0 | arm_style(1)=1,
  * reward terms wheel_speed, leg_symmetry, grounded, forward_lean, arms_back, skating_air_time exist and
    skating_air_time has weight 0 before stage 4,
  * the actuator plant respects the current/torque limit and the friction budget is non-zero,
  * a few random steps run without NaN.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import wheel_humanoid_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from wheel_humanoid_lab.actuators import CubeMarsActuator  # noqa: E402
from wheel_humanoid_lab.assets import (  # noqa: E402
    ACTOR_OBS_DIM,
    CMD_DIM,
    PROPRIO_DIM,
    Q1_JOINT_ORDER,
    Q1_PASSIVE_JOINTS,
    Q1_POSITION_JOINTS,
    Q1_WHEEL_JOINTS,
)

failures: list[str] = []


def check(cond: bool, msg: str):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main():
    env_cfg = parse_env_cfg("Isaac-Q1-Skate-v0", device="cuda:0", num_envs=args.num_envs)
    env = gym.make("Isaac-Q1-Skate-v0", cfg=env_cfg).unwrapped
    robot = env.scene["robot"]

    print("\n[joints]")
    check(set(Q1_JOINT_ORDER) <= set(robot.joint_names), "all 24 contract joints exist in the articulation")
    check(set(Q1_PASSIVE_JOINTS) <= set(robot.joint_names), "knee roller joints exist")
    check(robot.num_joints == 26, f"articulation has 26 joints (24 actuated + 2 rollers), got {robot.num_joints}")
    ids, names = robot.find_joints(Q1_JOINT_ORDER, preserve_order=True)
    check(names == Q1_JOINT_ORDER, "find_joints(preserve_order=True) returns the contract order")

    print("\n[actuators]")
    driven: dict[str, str] = {}
    for gname, act in robot.actuators.items():
        for j in act.joint_names:
            driven[j] = gname
        if isinstance(act, CubeMarsActuator):
            mode = act.cfg.control_mode
            for j in act.joint_names:
                if j in Q1_WHEEL_JOINTS:
                    check(mode == "velocity", f"{j} in velocity mode ({gname})")
                else:
                    check(mode == "position", f"{j} in position mode ({gname})")
            check(float(act.friction.max()) == 0.0, f"{gname}: PhysX joint friction is 0 (friction in actuator)")
            check(act.cfg.coulomb_friction > 0.0, f"{gname}: actuator friction budget > 0")
            check(float(act.effort_limit.max()) <= act.cfg.peak_torque + 1e-6, f"{gname}: effort limit == SKU peak torque {act.cfg.peak_torque} Nm")
    for j in Q1_JOINT_ORDER:
        check(isinstance(robot.actuators.get(driven.get(j, ""), None), CubeMarsActuator), f"{j} driven by CubeMarsActuator ({driven.get(j)})")
    skus = {a.cfg.sku for a in robot.actuators.values() if isinstance(a, CubeMarsActuator)}
    check(skus == {"AKE90-8_KV35", "AK10-9_V3_KV60", "AK45-36_V3_KV80", "AK45-10_V3_KV75"}, f"4 CubeMars SKUs grouped: {sorted(skus)}")
    print(f"  total mass (env 0): {float(robot.data.default_mass[0].sum()):.2f} kg")

    print("\n[observation / action contract]")
    obs, _ = env.reset()
    pol = obs["policy"]
    crit = obs["critic"]
    check(pol.shape[1] == ACTOR_OBS_DIM, f"policy obs dim {pol.shape[1]} == {ACTOR_OBS_DIM} (proprio {PROPRIO_DIM} + cmd {CMD_DIM})")
    check(crit.shape[1] > pol.shape[1], f"critic obs is privileged/wider ({crit.shape[1]} > {pol.shape[1]})")
    check(env.action_manager.total_action_dim == 24, f"action dim {env.action_manager.total_action_dim} == 24")
    check(env.action_manager.get_term("joint_pos")._joint_names == Q1_POSITION_JOINTS, "position action term is in contract order (22)")
    check(env.action_manager.get_term("wheel_vel")._joint_names == Q1_WHEEL_JOINTS, "wheel velocity term = [l_wheel, r_wheel]")
    cmd = pol[:, PROPRIO_DIM:]
    check(torch.all(cmd[:, 3:13] == 0.0).item(), "head(4) + body(6) command slots are zero padded")
    check(torch.all(cmd[:, 13] == 1.0).item(), "arm_style slot == 1 (arms-back skate)")
    terms = env.observation_manager.active_terms["policy"]
    check(terms == ["imu_gyro", "imu_projected_gravity", "joint_pos", "joint_vel", "last_action", "command"], f"policy term order {terms}")

    print("\n[rewards / curriculum]")
    rnames = env.reward_manager.active_terms
    for r in ["wheel_speed", "leg_symmetry", "grounded", "forward_lean", "arms_back", "skating_air_time", "wheel_slip", "flying", "long_air", "heading_hold"]:
        check(r in rnames, f"reward term '{r}' registered")
    check(env.reward_manager.get_term_cfg("skating_air_time").weight == 0.0, "skating_air_time weight is 0 before stage 4")
    check(env.reward_manager.get_term_cfg("forward_lean").weight == 0.0, "forward_lean weight is 0 before stage 3")
    check("stage" in env.curriculum_manager.active_terms, "stage curriculum registered")

    print("\n[plant sanity: random actions]")
    nan = False
    sat_seen = False
    for _ in range(50):
        a = torch.rand(env.num_envs, 24, device=env.device) * 2.0 - 1.0
        obs, rew, term, trunc, info = env.step(a)
        nan |= bool(torch.isnan(obs["policy"]).any() or torch.isnan(rew).any())
        hip = robot.actuators["ake90_hip"]
        sat_seen |= bool(hip.saturated.any())
        check_lim = float(robot.data.applied_torque[:, hip.joint_indices].abs().max()) <= hip.cfg.peak_torque + 1e-4
        if not check_lim:
            failures.append("AKE90 applied torque exceeded the peak")
            break
    check(not nan, "no NaN in obs/rewards after 50 random steps")
    check(float(robot.data.applied_torque.abs().max()) <= 121.0 + 1e-4, "no joint exceeds the largest SKU peak torque (121 Nm)")
    hip = robot.actuators["ake90_hip"]
    print(f"  AKE90 hip: |i| max {float(hip.current.abs().max()):.1f} A (limit {hip.cfg.peak_current} A), "
          f"|tau| max {float(hip.applied_effort.abs().max()):.1f} Nm, friction |tau_f| mean {float(hip.friction_torque.abs().mean()):.2f} Nm, "
          f"bus V range [{float(hip.bus_voltage.min()):.1f}, {float(hip.bus_voltage.max()):.1f}], saturated seen: {sat_seen}")
    wh = robot.actuators["ak10_wheels"]
    print(f"  AK10 wheels: |omega| max {float(robot.data.joint_vel[:, wh.joint_indices].abs().max()):.1f} rad/s, |tau| max {float(wh.applied_effort.abs().max()):.1f} Nm")

    env.close()
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES:\n  - " + "\n  - ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    rc = main()
    app.close()
    sys.exit(rc)
