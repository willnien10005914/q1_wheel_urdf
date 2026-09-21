"""Isaac Lab ArticulationCfg template (Isaac Lab 2.x — check API names for your version).

Gains, torque, and speed limits are provisional.
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "wheel_humanoid.urdf")

WHEEL_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=1.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Convex hulls overlap non-adjacent links (waist, knee rollers), so disable self-collision.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Zero pose: wheel bottoms sit 0.8867 m below the pelvis origin.
        pos=(0.0, 0.0, 0.90),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=60.0, velocity_limit_sim=12.0,
            stiffness=80.0, damping=4.0,
        ),
        "wheels": ImplicitActuatorCfg(  # velocity control: stiffness=0
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=20.0, velocity_limit_sim=30.0,
            stiffness=0.0, damping=2.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=40.0, velocity_limit_sim=6.0,
            stiffness=60.0, damping=3.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
            effort_limit_sim=30.0, velocity_limit_sim=10.0,
            stiffness=40.0, damping=2.0,
        ),
        # *_knee_roller_joint uses URDF <mimic> (q = 0.444 * q_knee).
        # Isaac Sim 4.5+ turns that into a PhysX mimic joint, so do not attach an actuator.
    },
)
