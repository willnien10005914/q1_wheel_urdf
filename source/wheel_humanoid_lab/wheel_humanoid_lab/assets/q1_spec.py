"""Q1 Wheel hardware sheet + the shared 24-D joint contract.

Everything here is name based. Never index PhysX joints by position: use ``Q1_JOINT_ORDER`` and
``Articulation.find_joints(..., preserve_order=True)``.
"""

from __future__ import annotations

import math
import os

import yaml

from wheel_humanoid_lab import WHEEL_HUMANOID_ROOT_DIR

COMPONENTS_YAML = os.path.join(WHEEL_HUMANOID_ROOT_DIR, "config", "q1_wheel_components.yaml")

with open(COMPONENTS_YAML) as _f:
    Q1_COMPONENTS: dict = yaml.safe_load(_f)

MOTORS: dict[str, dict] = Q1_COMPONENTS["motors"]
JOINT_MOTOR: dict[str, dict] = Q1_COMPONENTS["joints"]
WHEEL_RADIUS: float = float(Q1_COMPONENTS["robot"]["wheel_radius_m"])
CAN_RATE_HZ: int = int(Q1_COMPONENTS["bus"]["can_rate_hz"])
POLICY_RATE_HZ: int = int(Q1_COMPONENTS["bus"]["policy_rate_hz"])
PHYSICS_DT: float = 1.0 / CAN_RATE_HZ
DECIMATION: int = CAN_RATE_HZ // POLICY_RATE_HZ
IMU_CFG: dict = Q1_COMPONENTS["imu"]

# ---------------------------------------------------------------------------------------------
# 24-D joint contract (also the action layout: position-controlled joints first, wheels last).
# The order is frozen: skate / stand / recover policies all share it so ONNX files can hot-swap.
# ---------------------------------------------------------------------------------------------
Q1_JOINT_ORDER: list[str] = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "neck_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_joint",
    "l_wrist_joint",
    "l_gripper_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_joint",
    "r_wrist_joint",
    "r_gripper_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_knee_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_knee_joint",
    "l_wheel_joint",
    "r_wheel_joint",
]
assert len(Q1_JOINT_ORDER) == 24 and set(Q1_JOINT_ORDER) == set(JOINT_MOTOR), "24-D contract out of sync with YAML"

Q1_WHEEL_JOINTS: list[str] = ["l_wheel_joint", "r_wheel_joint"]
Q1_POSITION_JOINTS: list[str] = [j for j in Q1_JOINT_ORDER if j not in Q1_WHEEL_JOINTS]
Q1_FROZEN_JOINTS: list[str] = [j for j, s in JOINT_MOTOR.items() if s.get("frozen", False)]
Q1_PASSIVE_JOINTS: list[str] = ["l_knee_roller_joint", "r_knee_roller_joint"]  # mimic linkage, not motors

# Mirror-symmetry sign convention for leg_symmetry (q_L - s * q_R = 0):
#  pitch/knee axes are +y on both sides -> same sign; roll/yaw axes flip -> opposite sign.
Q1_LEG_PAIRS: list[tuple[str, str, float]] = [
    ("l_hip_pitch_joint", "r_hip_pitch_joint", 1.0),
    ("l_hip_roll_joint", "r_hip_roll_joint", -1.0),
    ("l_knee_joint", "r_knee_joint", 1.0),
]
Q1_ARM_PAIRS: list[tuple[str, str, float]] = [
    ("l_shoulder_pitch_joint", "r_shoulder_pitch_joint", 1.0),
    ("l_shoulder_roll_joint", "r_shoulder_roll_joint", -1.0),
    ("l_shoulder_yaw_joint", "r_shoulder_yaw_joint", -1.0),
    ("l_elbow_joint", "r_elbow_joint", 1.0),
]

# Command block of the observation contract (widths are frozen; unused slots are zero padded).
CMD_TWIST_DIM = 3  # vx, vy(=0), yaw
CMD_HEAD_DIM = 4
CMD_BODY_DIM = 6
CMD_ARM_STYLE_DIM = 1  # 0 = neutral, 1 = arms-back skate
CMD_DIM = CMD_TWIST_DIM + CMD_HEAD_DIM + CMD_BODY_DIM + CMD_ARM_STYLE_DIM
PROPRIO_DIM = 3 + 3 + 24 + 24 + 24
ACTOR_OBS_DIM = PROPRIO_DIM + CMD_DIM  # 78 + 14 = 92


def motor_of(joint: str) -> dict:
    return MOTORS[JOINT_MOTOR[joint]["motor"]]


def no_load_speed_rad_s(sku: dict) -> float:
    return sku["no_load_speed_rpm"] * 2.0 * math.pi / 60.0


def reflected_rotor_inertia(sku: dict) -> float:
    """Rotor inertia seen at the output: J_rotor * G^2 (used as PhysX armature)."""
    return sku["rotor_inertia_kgm2"] * sku["gear_ratio"] ** 2
