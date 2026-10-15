"""Native BU10 wheeled asset: original CAD physics and a separate 23-D policy contract.

The updated source has no neck joint. Joint axes are kept exactly as supplied, so
left wheel forward velocity is negative and right wheel forward velocity is positive.
See config/q1_inertia_manifest.json for provenance and FK measurements.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from wheel_humanoid_lab import WHEEL_HUMANOID_ROOT_DIR
from .wheel_humanoid import _cubemars, _FRIC_AKE90, _FRIC_AK10, _FRIC_AK10_WHEEL, _FRIC_AK45_10


class SourceInertiaUrdfConverter(UrdfConverter):
    """Pin CAD inertia import for this asset instead of relying on importer defaults."""

    def _get_urdf_import_config(self):
        import_config = super()._get_urdf_import_config()
        import_config.set_import_inertia_tensor(self.cfg.import_inertia_tensor)
        return import_config


@clone
def spawn_source_inertia_urdf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    converter = SourceInertiaUrdfConverter(cfg)
    return _spawn_from_usd_file(prim_path, converter.usd_path, cfg, translation, orientation, **kwargs)


@configclass
class SourceInertiaUrdfFileCfg(sim_utils.UrdfFileCfg):
    func: Callable = spawn_source_inertia_urdf
    # Included in the converter cache hash, so a USD generated with a different
    # inertia import policy cannot be reused silently.
    import_inertia_tensor: bool = True


Q1_INERTIA_URDF = os.path.join(WHEEL_HUMANOID_ROOT_DIR, "urdf", "q1_inertia", "quanta_robot_wheeled.urdf")
Q1_INERTIA_WHEEL_RADIUS = 0.1
Q1_INERTIA_PHYSICS_DT = 0.005
Q1_INERTIA_DECIMATION = 4
Q1_INERTIA_ROOT_Z = 0.8466859904792701
Q1_INERTIA_POSITION_JOINTS = [
    "center_waist_Yaw_joint", "center_waist_Roll_joint", "center_waist_Pitch_joint",
    "left_shoulder_Pitch_joint", "left_shoulder_Roll_joint", "left_shoulder_Yaw_joint",
    "left_elbow_Pitch_joint", "left_wrist_Yaw_joint", "left_finger_1_Roll_joint",
    "right_shoulder_Pitch_joint", "right_shoulder_Roll_joint", "right_shoulder_Yaw_joint",
    "right_elbow_Pitch_joint", "right_wrist_Yaw_joint", "right_finger_1_Roll_joint",
    "left_hip_Pitch_joint", "left_hip_Roll_joint", "left_knee_1_Pitch_joint",
    "right_hip_Pitch_joint", "right_hip_Roll_joint", "right_knee_1_Pitch_joint",
]
Q1_INERTIA_WHEEL_JOINTS = ["left_wheel_Pitch_joint", "right_wheel_Pitch_joint"]
Q1_INERTIA_JOINT_ORDER = Q1_INERTIA_POSITION_JOINTS + Q1_INERTIA_WHEEL_JOINTS
Q1_INERTIA_MIMIC_JOINTS = [
    "left_knee_2_Pitch_joint", "right_knee_2_Pitch_joint",
    "left_finger_2_Roll_joint", "right_finger_2_Roll_joint",
]
Q1_INERTIA_MIMIC_DRIVERS = [
    "left_knee_1_Pitch_joint", "right_knee_1_Pitch_joint",
    "left_finger_1_Roll_joint", "right_finger_1_Roll_joint",
]
Q1_INERTIA_MIMIC_MULTIPLIERS = [0.444444, 0.444444, -1.0, -1.0]
Q1_INERTIA_DEFAULT_POS = {joint: 0.0 for joint in Q1_INERTIA_JOINT_ORDER + Q1_INERTIA_MIMIC_JOINTS}
Q1_INERTIA_DEFAULT_POS.update({
    "center_waist_Pitch_joint": 0.10,
    "left_shoulder_Pitch_joint": 0.55, "right_shoulder_Pitch_joint": -0.55,
    "left_shoulder_Roll_joint": 0.20, "right_shoulder_Roll_joint": -0.20,
    "left_elbow_Pitch_joint": -0.50, "right_elbow_Pitch_joint": 0.50,
    "left_hip_Pitch_joint": 0.35, "right_hip_Pitch_joint": -0.35,
    "left_knee_1_Pitch_joint": -0.70, "right_knee_1_Pitch_joint": 0.70,
    "left_knee_2_Pitch_joint": -0.70 * 0.444444,
    "right_knee_2_Pitch_joint": 0.70 * 0.444444,
})
# Exported runtime contract: actor = gyro3 + gravity3 + command3 + q23 + dq23 + previous_action23.
Q1_INERTIA_ACTION_DIM = 23
Q1_INERTIA_ACTOR_OBS_DIM = 78
Q1_INERTIA_CRITIC_OBS_DIM = 81

# Catch source drift before creating a simulator or silently reordering policy channels.
_SOURCE_JOINTS = ET.parse(Q1_INERTIA_URDF).getroot().findall("joint")
assert {j.get("name") for j in _SOURCE_JOINTS if j.find("mimic") is None} == set(Q1_INERTIA_JOINT_ORDER)
assert {j.get("name") for j in _SOURCE_JOINTS if j.find("mimic") is not None} == set(Q1_INERTIA_MIMIC_JOINTS)

Q1_INERTIA_CFG = ArticulationCfg(
    spawn=SourceInertiaUrdfFileCfg(
        asset_path=Q1_INERTIA_URDF,
        usd_dir=os.path.join(WHEEL_HUMANOID_ROOT_DIR, "usd", "q1_inertia"),
        usd_file_name="quanta_robot_wheeled.usd",
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=True,
        make_instanceable=True,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, Q1_INERTIA_ROOT_Z + 0.002),
        joint_pos=Q1_INERTIA_DEFAULT_POS, joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "ake90_legs": _cubemars(
            "AKE90-8_KV35", ["(left|right)_hip_(Pitch|Roll)_joint", "(left|right)_knee_1_Pitch_joint"],
            mode="position", kp=180.0, kd=7.0, fric=_FRIC_AKE90, sag=0.015,
        ),
        "ak10_waist": _cubemars(
            "AK10-9_V3_KV60", ["center_waist_.*_joint"],
            mode="position", kp=70.0, kd=3.0, fric=_FRIC_AK10, sag=0.02,
        ),
        "ak10_arms": _cubemars(
            "AK10-9_V3_KV60", ["(left|right)_shoulder_.*_joint", "(left|right)_elbow_Pitch_joint"],
            mode="position", kp=45.0, kd=2.0, fric=_FRIC_AK10, sag=0.02,
        ),
        "ak10_wheels": _cubemars(
            "AK10-9_V3_KV60", Q1_INERTIA_WHEEL_JOINTS,
            mode="velocity", kp=0.0, kd=1.2, fric=_FRIC_AK10_WHEEL, sag=0.02,
        ),
        "ak45_hands": _cubemars(
            "AK45-10_V3_KV75", ["(left|right)_wrist_Yaw_joint", "(left|right)_finger_1_Roll_joint"],
            mode="position", kp=8.0, kd=0.4, fric=_FRIC_AK45_10, sag=0.05,
        ),
        # Source linkage tags are retained in URDF; this finite-gain servo approximation
        # prevents importer-dependent mimic behavior from changing the 23-D policy layout.
        "mimic_knees": ImplicitActuatorCfg(
            joint_names_expr=Q1_INERTIA_MIMIC_JOINTS[:2],
            effort_limit_sim=100.0, velocity_limit_sim=10.0, stiffness=300.0, damping=8.0,
        ),
        "mimic_fingers": ImplicitActuatorCfg(
            joint_names_expr=Q1_INERTIA_MIMIC_JOINTS[2:],
            effort_limit_sim=7.0, velocity_limit_sim=10.0, stiffness=30.0, damping=1.0,
        ),
    },
)
