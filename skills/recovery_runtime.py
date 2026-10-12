"""Run each recovery checkpoint with its own immutable contact-controller snapshot."""
import importlib.util,sys,json,hashlib
import re
from pathlib import Path
from types import SimpleNamespace
import torch
from isaaclab.managers import ObservationTermCfg,SceneEntityCfg
from wheel_humanoid_lab.assets import Q1_JOINT_ORDER,Q1_WHEEL_CUBEMARS_CFG
from wheel_humanoid_lab.assets.wheel_humanoid import SKATE_HIP_PITCH,SKATE_KNEE
from wheel_humanoid_lab.tasks.manager_based.recovery.recovery_env_cfg import RecoveryObservationsCfg
from wheel_humanoid_lab.tasks.manager_based.skate.mdp.observations import ImuObs,JointPosContract,JointVelContract
from skills.policy import FrozenActor

class Context:
    def __init__(self,env):
        self.env=env
        self.scene=SceneProxy(env)
    def __getattr__(self,name):return getattr(self.env,name)

class SceneProxy:
    def __init__(self,env):self.env=env
    def __getitem__(self,name):
        return self.env.contacts if name=='contact_forces' else self.env.scene[name]
    def __getattr__(self,name):return getattr(self.env.scene,name)

class RecoveryRuntime:
    def __init__(self,env,root,variant,checkpoint):
        self.env=env;self.root=Path(root);self.variant=variant
        snapshot=self.root/'skills/recovery'/variant
        metadata=snapshot/'runtime_metadata.json'
        if metadata.is_file():
            proof=json.loads(metadata.read_text())
            if hashlib.sha256((self.root/'checkpoints'/checkpoint).read_bytes()).hexdigest()!=proof['checkpoint_sha256']:
                raise RuntimeError('Recovery checkpoint differs from validation: '+variant)
            for name,expected in proof['files_sha256'].items():
                if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=expected:
                    raise RuntimeError('Recovery controller differs from validation: '+variant+' / '+name)
        source=snapshot/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/recovery_v3/mdp.py'
        spec=importlib.util.spec_from_file_location('q1_snapshot_'+variant,source)
        self.mdp=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=self.mdp
        spec.loader.exec_module(self.mdp)
        self.mdp.WHEEL_HUMANOID_ROOT_DIR=str(snapshot)
        self.context=Context(env)
        self.state=self.mdp.state(self.context)
        self.state.cap=4
        self.actor=FrozenActor(self.root/'checkpoints'/checkpoint,env.device)
        self.pos_ids=self.state.pos_ids
        self.ids=env.robot.find_joints(Q1_JOINT_ORDER,preserve_order=True)[0]
        # Isaac clamps the nominal zero gripper pose to its physical lower
        # limit (~0.03 rad). Keep that actual encoder reference: reconstructing
        # all defaults from nominal asset values corrupts near-constant channels.
        self.default=env.robot.data.default_joint_pos[:,self.ids].clone()
        for i,j in enumerate(self.ids):
            name=env.robot.joint_names[j]
            if name.endswith('_hip_pitch_joint'):self.default[:,i]=SKATE_HIP_PITCH
            if name.endswith('_knee_joint'):self.default[:,i]=SKATE_KNEE
        self.last_action=torch.zeros(env.num_envs,24,device=env.device)
        # Retain the training observation implementations (IMU mounting/delay,
        # encoders, delayed velocity); only command/default-pose mapping differs.
        cfg=RecoveryObservationsCfg.PolicyCfg()
        self.gyro=ImuObs(cfg.imu_gyro,self.context)
        self.gravity=ImuObs(cfg.imu_projected_gravity,self.context)
        self.positions=JointPosContract(cfg.joint_pos,self.context)
        self.velocities=JointVelContract(cfg.joint_vel,self.context)
        self.position_term=SimpleNamespace(_env=self.context,_asset=env.robot,_joint_ids=self.pos_ids,
            _raw_actions=torch.zeros(env.num_envs,len(self.pos_ids),device=env.device))
        self.running=False

    def _reset_observations(self):
        self.gyro.reset();self.gravity.reset();self.positions.reset();self.velocities.reset();self.last_action.zero_()

    def place_lie(self,mode):
        self.mdp.reset(self.context,torch.arange(self.env.num_envs,device=self.env.device),mode=mode)
        self._reset_observations();self.running=False

    def start_floor(self):
        # Continue from the selected floor pose, without rewriting simulator state.
        self.running=True;self.state.cap=4

    def start_kneel(self):
        s=self.state;n=self.env.num_envs
        # Same stage initialization as boot_kneel reset, using the CURRENT pose.
        # No root, joint, or velocity state writes at this handover.
        for name in ['elapsed','hold','stand_hold','upright_age','upright_hold']:
            getattr(s,name).zero_()
        s.mode.zero_();s.stage.fill_(3);s.boot_assist.fill_(True)
        s.knelt.fill_(True);s.planted.fill_(True);s.kneel_hold.fill_(1.)
        s.upright_knelt.fill_(True);s.cap=4;s.stand_allowed.fill_(True)
        s.stood.fill_(False);s.upright_captured.fill_(False)
        q=self.env.robot.data.joint_pos.clone()
        s.start_q.copy_(q);s.command.copy_(q);s.held_command.copy_(q)
        s.transfer_held.fill_(True);s.upright_start.copy_(q);s.upright_command.copy_(q);s.upright_active.fill_(True)
        s.filtered_force.zero_();s.last_step=-1;s.dirty=True
        self._reset_observations();self.running=True

    def start_landing(self):
        self.start_kneel()
        s=self.state
        s.stage.fill_(4);s.elapsed.zero_();s.hold.zero_()
        s.upright_active.fill_(False);s.dirty=True

    def observation(self):
        cfg=RecoveryObservationsCfg.PolicyCfg()
        gyro=self.gyro(self.context,**cfg.imu_gyro.params)
        gravity=self.gravity(self.context,**cfg.imu_projected_gravity.params)
        q=self.positions(self.context,**cfg.joint_pos.params)
        actual_default=self.env.robot.data.default_joint_pos[:,self.ids]
        q=q+actual_default-self.default
        wheel=torch.tensor([self.env.robot.joint_names[j].endswith('_wheel_joint') for j in self.ids],device=self.env.device)
        q=torch.where(wheel,(q+torch.pi)%(2*torch.pi)-torch.pi,q)
        velocity=self.velocities(self.context,**cfg.joint_vel.params)
        return torch.cat((gyro,gravity,q,velocity,self.last_action,self.mdp.observation(self.context)),dim=-1)

    def act(self):
        action=self.actor(self.observation())
        # Invoke the exact training position processor, including all stage gates.
        self.mdp.ContactPositionAction.process_actions(self.position_term,action[:,:22])
        position=self.position_term._processed_actions
        residual=(8.*action[:,22:]).clamp(-8,8)
        residual=torch.where((self.state.stage==3)[:,None],residual.clamp(-1.5,1.5),residual)
        wheel=(residual+self.mdp.balance_velocity(self.state).clamp(-25,25)[:,None]).clamp(-25,25)
        self.last_action.copy_(action)
        return position,wheel
