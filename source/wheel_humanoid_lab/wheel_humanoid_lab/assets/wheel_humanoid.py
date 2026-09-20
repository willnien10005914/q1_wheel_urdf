"""Articulation configs for the Q1 / BI2 dual-wheel humanoid.

Two plants are provided:

* ``WHEEL_HUMANOID_CFG`` / ``WHEEL_HUMANOID_SKATEBOARD_CFG``: legacy ideal implicit PD (kept for the
  old ``Isaac-WheelHumanoid-Skateboard-v0`` task and for quick viewer checks).
* ``Q1_WHEEL_CUBEMARS_CFG``: 24 CubeMars actuators grouped by SKU with the explicit
  :class:`CubeMarsActuator` plant (voltage/current limits, in-actuator friction, delay, sag, DR).
  This is what the skate task trains on.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from wheel_humanoid_lab import WHEEL_HUMANOID_ROOT_DIR
from wheel_humanoid_lab.actuators import CubeMarsActuatorCfg

from .q1_spec import MOTORS, Q1_COMPONENTS, reflected_rotor_inertia

URDF_PATH = os.path.join(WHEEL_HUMANOID_ROOT_DIR, "urdf", "wheel_humanoid.urdf")
USD_DIR = os.path.join(WHEEL_HUMANOID_ROOT_DIR, "usd")

# Standing skate pose: only the two foot wheels contact the ground, wheel axle right under the hip.
# hip -0.35 / knee 0.70 rad -> pelvis 0.847 m, knee rollers 0.30 m above ground (tools FK check).
SKATE_PELVIS_Z = 0.847
SKATE_HIP_PITCH = -0.35
SKATE_KNEE = 0.70
SKATE_WAIST_PITCH = 0.10
# Arms-back skate keyframe (X2 clip docs/reference: arms trail behind the torso, elbows slightly bent).
SKATE_SHOULDER_PITCH = 0.55
SKATE_SHOULDER_ROLL = 0.20  # left +, right - (abduction)
SKATE_ELBOW = -0.50

_BUS = Q1_COMPONENTS["bus"]
_PACK_DR = tuple(_BUS["pack_voltage_dr_v"])
_RAIL24_DR = tuple(_BUS["rail_24v_dr_v"])


def _cubemars(sku_name: str, joints: list[str], *, mode: str, kp: float, kd: float, fric: dict, sag: float):
    sku = MOTORS[sku_name]
    v_dr = _PACK_DR if sku["bus_voltage_v"] >= 40.0 else _RAIL24_DR
    return CubeMarsActuatorCfg(
        joint_names_expr=joints,
        sku=sku_name,
        control_mode=mode,
        stiffness=kp,
        damping=kd,
        armature=reflected_rotor_inertia(sku),
        bus_voltage=sku["bus_voltage_v"],
        bus_voltage_range=v_dr,
        kv_rpm_per_v=sku["kv_rpm_per_v"],
        kt_nm_per_a=sku["kt_nm_per_a"],
        phase_resistance=sku["phase_resistance_ohm"],
        gear_ratio=sku["gear_ratio"],
        gear_efficiency=0.90,
        peak_current=sku["peak_current_a"],
        cont_current=sku["rated_current_a"],
        peak_torque=sku["tau_peak_slide_nm"],
        cont_torque=sku["rated_torque_nm"],
        coulomb_friction=fric["c"],
        stribeck_friction=fric["s"],
        stribeck_velocity=0.15,
        viscous_friction=fric["b"],
        load_dep_friction=0.05,
        friction_scale_range=(0.85, 1.15),
        min_delay=0,
        max_delay=3,
        voltage_sag_gain=sag,
        voltage_min=0.8 * sku["bus_voltage_v"],
    )


# Friction budgets on the output side (Coulomb, Stribeck break-away, viscous). AKE90 is a 5-stage
# planetary; AK45-36 lists 0.8 Nm back-drive torque; wheels include rolling resistance of the tyre.
_FRIC_AKE90 = {"c": 0.8, "s": 0.4, "b": 0.05}
_FRIC_AK10 = {"c": 0.4, "s": 0.2, "b": 0.02}
_FRIC_AK10_WHEEL = {"c": 0.6, "s": 0.1, "b": 0.03}
_FRIC_AK45_36 = {"c": 0.5, "s": 0.3, "b": 0.01}
_FRIC_AK45_10 = {"c": 0.1, "s": 0.05, "b": 0.005}

_SPAWN = sim_utils.UrdfFileCfg(
    asset_path=URDF_PATH,
    usd_dir=USD_DIR,
    usd_file_name="wheel_humanoid.usd",
    fix_base=False,
    merge_fixed_joints=False,
    make_instanceable=True,
    activate_contact_sensors=True,
    replace_cylinders_with_capsules=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    ),
)

SKATE_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, SKATE_PELVIS_Z),
    joint_pos={
        ".*_hip_pitch_joint": SKATE_HIP_PITCH,
        ".*_hip_roll_joint": 0.0,
        ".*_knee_joint": SKATE_KNEE,
        ".*_knee_roller_joint": 0.444444 * SKATE_KNEE,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": SKATE_WAIST_PITCH,
        "neck_joint": 0.0,
        "l_shoulder_pitch_joint": SKATE_SHOULDER_PITCH,
        "r_shoulder_pitch_joint": SKATE_SHOULDER_PITCH,
        "l_shoulder_roll_joint": SKATE_SHOULDER_ROLL,
        "r_shoulder_roll_joint": -SKATE_SHOULDER_ROLL,
        ".*_shoulder_yaw_joint": 0.0,
        ".*_elbow_joint": SKATE_ELBOW,
        ".*_wrist_joint": 0.0,
        ".*_gripper_joint": 0.0,
        ".*_wheel_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Legacy ideal-PD plant (old skateboard task).
##

WHEEL_HUMANOID_CFG = ArticulationCfg(
    spawn=_SPAWN,
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.90), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=121.0,
            velocity_limit_sim=22.0,
            stiffness=150.0,
            damping=6.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=43.0,
            velocity_limit_sim=33.5,
            stiffness=0.0,
            damping=2.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=43.0,
            velocity_limit_sim=33.5,
            stiffness=60.0,
            damping=3.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
            effort_limit_sim=43.0,
            velocity_limit_sim=33.5,
            stiffness=40.0,
            damping=2.0,
        ),
        "small": ImplicitActuatorCfg(
            joint_names_expr=["neck_joint", ".*_wrist_joint", ".*_gripper_joint"],
            effort_limit_sim=7.0,
            velocity_limit_sim=18.0,
            stiffness=10.0,
            damping=0.5,
        ),
        "rollers": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_roller_joint"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=5.0,
            damping=0.3,
        ),
    },
)

WHEEL_HUMANOID_SKATEBOARD_CFG = WHEEL_HUMANOID_CFG.replace(init_state=SKATE_INIT_STATE)

##
# CubeMars plant (skate task). Gains are joint-side and scaled with the SKU peak torque so the
# XL330-class gains of Microduck are not copied onto a 121 Nm actuator.
##

Q1_WHEEL_CUBEMARS_CFG = ArticulationCfg(
    spawn=_SPAWN,
    init_state=SKATE_INIT_STATE,
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "ake90_hip": _cubemars(
            "AKE90-8_KV35", [".*_hip_pitch_joint", ".*_hip_roll_joint"], mode="position", kp=180.0, kd=7.0,
            fric=_FRIC_AKE90, sag=0.015,
        ),
        "ake90_knee": _cubemars(
            "AKE90-8_KV35", [".*_knee_joint"], mode="position", kp=180.0, kd=7.0, fric=_FRIC_AKE90, sag=0.015
        ),
        "ak10_torso": _cubemars(
            "AK10-9_V3_KV60", ["waist_.*_joint"], mode="position", kp=70.0, kd=3.0, fric=_FRIC_AK10, sag=0.02
        ),
        "ak10_arms": _cubemars(
            "AK10-9_V3_KV60", [".*_shoulder_.*_joint", ".*_elbow_joint"], mode="position", kp=45.0, kd=2.0,
            fric=_FRIC_AK10, sag=0.02,
        ),
        "ak10_wheels": _cubemars(
            "AK10-9_V3_KV60", [".*_wheel_joint"], mode="velocity", kp=0.0, kd=1.2, fric=_FRIC_AK10_WHEEL, sag=0.02
        ),
        "ak45_neck": _cubemars(
            "AK45-36_V3_KV80", ["neck_joint"], mode="position", kp=15.0, kd=0.8, fric=_FRIC_AK45_36, sag=0.05
        ),
        "ak45_wrist_gripper": _cubemars(
            "AK45-10_V3_KV75", [".*_wrist_joint", ".*_gripper_joint"], mode="position", kp=8.0, kd=0.4,
            fric=_FRIC_AK45_10, sag=0.05,
        ),
        # knee rollers are a mimic linkage on the real robot (0.444 * knee); the URDF importer does not
        # create the constraint, so a weak implicit spring keeps them near the linkage angle.
        "passive_rollers": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_roller_joint"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=5.0,
            damping=0.3,
        ),
    },
)
