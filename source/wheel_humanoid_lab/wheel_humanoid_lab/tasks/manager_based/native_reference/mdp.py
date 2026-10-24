"""MediaPipe-reference managers with contact-aware motor-only residual control."""
from __future__ import annotations
from pathlib import Path
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from ..inertia import native_skills as base
from ..native_slide.mdp import SlideCommand
from ..native_slide.gait import profile, measured_rewards
from wheel_humanoid_lab import WHEEL_HUMANOID_ROOT_DIR
from wheel_humanoid_lab.assets.q1_inertia import (
    Q1_INERTIA_JOINT_ORDER, Q1_INERTIA_POSITION_JOINTS, Q1_INERTIA_URDF,
    Q1_INERTIA_MIMIC_JOINTS, Q1_INERTIA_MIMIC_DRIVERS, Q1_INERTIA_MIMIC_MULTIPLIERS,
)
from .trajectory import VideoTrajectory, RELATIVE_REFERENCE, CONTRACT
from .profiles import backflip_reference, general_leg_angles


class ReferenceCommand(SlideCommand):
    def __init__(self,cfg,env):
        self.trajectory=VideoTrajectory(Path(WHEEL_HUMANOID_ROOT_DIR)/cfg.reference_path,
                                        Q1_INERTIA_JOINT_ORDER,Q1_INERTIA_URDF,device=env.device)
        super().__init__(cfg,env)
        self.indices={name:i for i,name in enumerate(Q1_INERTIA_POSITION_JOINTS)}
        self.poses['stand']=self.trajectory.q[-1,:21].clone()
        self.poses['kneel']=self.trajectory.q[0,:21].clone()
        self.phase_offset=torch.zeros(self.num_envs,device=self.device)
        self.pending_reset_phase=self.phase_offset.clone()
        self.fixed_phase=torch.full_like(self.phase_offset,-1.)
        self.ref_hold=self.phase_offset.clone();self.ref_best_hold=self.phase_offset.clone()
        self.ref_success=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.touchdown_t=self.phase_offset.clone();self.early_air=torch.zeros_like(self.ref_success)
        self.support_body_ids,_=self.robot.find_bodies(self.support_names,preserve_order=True)
        self.support_sensor_ids,_=self.sensor.find_bodies(self.support_names,preserve_order=True)
        self.arm_indices=[self.indices[n] for n in Q1_INERTIA_POSITION_JOINTS if 'shoulder_' in n or 'elbow_' in n]

    @property
    def normalized_phase(self):
        phase=((self.elapsed+self.phase_offset)/self.trajectory.duration).clamp(0.,1.)
        return torch.where(self.fixed_phase>=0,self.fixed_phase,phase)

    @property
    def phase(self):return self.normalized_phase

    def _ids(self,env_ids):
        return self.robot._ALL_INDICES if env_ids is None or isinstance(env_ids,slice) else env_ids

    def set_mode(self,name,env_ids=None):
        super().set_mode(name,env_ids)
        if not hasattr(self,'phase_offset'):return
        ids=self._ids(env_ids)
        self.phase_offset[ids]=0.;self.fixed_phase[ids]=-1.
        self.ref_hold[ids]=0.;self.ref_best_hold[ids]=0.;self.ref_success[ids]=False
        self.touchdown_t[ids]=0.;self.early_air[ids]=False
        self.tick=-1;self._update_command()

    def hold_pose(self,kind,env_ids=None):
        if kind not in ('kneel','stand'):raise ValueError('Hold pose must be kneel or stand')
        self.set_mode('kneel' if kind=='kneel' else 'standup',env_ids)
        ids=self._ids(env_ids);self.fixed_phase[ids]=1.;self._update_command()

    def _resample_command(self,env_ids):
        self.set_mode(self.cfg.mode,env_ids)
        if not hasattr(self,'phase_offset'):return
        ids=self._ids(env_ids)
        # Isaac resets episode_length_buf AFTER the command manager. Recording
        # its old episode length here would freeze the next episode at phase 0.
        self.start_step[ids]=0.
        self.phase_offset[ids]=self.pending_reset_phase[ids]*self.trajectory.duration
        if self.cfg.boot_squat:
            self.mode[ids]=base.MODES.index('kneel');self.fixed_phase[ids]=1.;self.sliding=False
        self._update_command()

    def _update_command(self):
        super()._update_command()
        if not hasattr(self,'phase_offset'):return
        if self.sliding:return
        timed=(self.mode==base.MODES.index('standup'))|(self.mode==base.MODES.index('kneel'))
        self._command[:,5]=torch.where(timed,self.normalized_phase,self._command[:,5])

    def target_height(self):
        _,up_height,_=self.trajectory.sample(self.normalized_phase)
        _,down_height,_=self.trajectory.sample(self.normalized_phase,reverse=True)
        stand_height=self.trajectory.height[-1].expand(self.num_envs)
        result=torch.where(self.mode==base.MODES.index('standup'),up_height,
                           torch.where(self.mode==base.MODES.index('kneel'),down_height,stand_height))
        crouch_height=self.trajectory.height[0].expand_as(stand_height)
        load_height=stand_height+(crouch_height-stand_height)*base.smooth((self.elapsed-.5)/.7)
        return torch.where((self.mode==base.MODES.index('backflip'))&~self.has_air,load_height,result)

    def reference(self):
        if not hasattr(self,'phase_offset'):return super().reference()
        stand=self.poses['stand'].expand(self.num_envs,-1)
        up,_,_=self.trajectory.sample(self.normalized_phase)
        down,_,_=self.trajectory.sample(self.normalized_phase,reverse=True)
        ref=torch.where((self.mode==base.MODES.index('standup'))[:,None],up[:,:21],stand)
        ref=torch.where((self.mode==base.MODES.index('kneel'))[:,None],down[:,:21],ref)
        if self.sliding:
            _,_,x,lift=profile(self.elapsed)
            velocity=self._env.command_manager.get_command('base_velocity')[:,0]
            blend=base.smooth(self.elapsed/1.8)*self.level*(velocity.abs()/.25).clamp(0,1)
            for i,side in enumerate(('left','right')):
                sign=1. if side=='left' else -1.
                h=self.poses['stand'][self.indices[side+'_hip_Pitch_joint']]*sign
                k=-self.poses['stand'][self.indices[side+'_knee_1_Pitch_joint']]*sign
                x0=.32*torch.sin(h)+.34*torch.sin(h-k)
                down0=.32*torch.cos(h)+.34*torch.cos(h-k)
                # Keep a finite reach margin at the retargeted micro-bent stand.
                reach=torch.sqrt((.659**2-down0**2).clamp_min(.0025))
                target_x=x0+x[:,i]*blend
                target_x=target_x.clamp(-reach,reach)
                hip,knee=general_leg_angles(target_x,down0-lift[:,i]*blend)
                ref[:,self.indices[side+'_hip_Pitch_joint']]=sign*hip
                ref[:,self.indices[side+'_knee_1_Pitch_joint']]=-sign*knee
        flip=self.mode==base.MODES.index('backflip')
        if bool(flip.any()):
            flip_ref=backflip_reference(self.elapsed,stand,self.poses['kneel'].expand_as(stand),
                self.has_air,self.takeoff_t,self.winding,self.landed,self.touchdown_t,self.indices,self.level)
            ref=torch.where(flip[:,None],flip_ref,ref)
        return self.start_q+(ref-self.start_q)*base.smooth(self.elapsed/.35)[:,None]

    def update_state(self):
        if self.tick==self._env.common_step_counter:return
        previous_landed=self.landed.clone()
        super().update_state()
        if not hasattr(self,'ref_hold'):return
        newly_landed=self.landed&~previous_landed
        self.touchdown_t[newly_landed]=self.elapsed[newly_landed]
        flip=self.mode==base.MODES.index('backflip')
        self.early_air|=flip&(self.elapsed<1.1)&(self.air_time>.04)
        r=self.robot.data;z=r.root_pos_w[:,2]-self._env.scene.env_origins[:,2]
        normal=self.sensor.data.net_forces_w[:,self.support_sensor_ids,2].clamp_min(0.)
        support=(normal>12.).sum(-1)
        wheels=self.sensor.data.net_forces_w[:,self.wheel_sensor_ids,2].clamp_min(0.)
        upright=-r.projected_gravity_b[:,2]
        still=(r.root_lin_vel_b.norm(dim=-1)<.25)&(r.root_ang_vel_b.norm(dim=-1)<.6)
        error=(r.joint_pos[:,self.pids]-self.reference()).square().mean(-1).sqrt()
        stable=(upright>.92)&still&(support>=2)&~self.illegal&(error<.22)
        terminal=(self.normalized_phase>=.999)
        trajectory_mode=(self.mode==base.MODES.index('standup'))|(self.mode==base.MODES.index('kneel'))
        achieved=stable&(z-self.target_height()).abs().lt(.065)&torch.where(trajectory_mode,terminal,True)
        cmd=self._env.command_manager.get_command('base_velocity')
        walk=self.mode==base.MODES.index('walk')
        moving_ok=(r.root_lin_vel_b[:,0]-cmd[:,0]).abs()<.18
        moving_ok&=(r.root_ang_vel_b[:,2]-cmd[:,2]).abs()<.2
        achieved=torch.where(walk,(upright>.92)&(support>=1)&~self.illegal&moving_ok&(z-self.target_height()).abs().lt(.08),achieved)
        arms=(r.joint_pos[:,self.pids][:,self.arm_indices]-self.poses['stand'][self.arm_indices]).abs().amax(-1)<.18
        circle=(self.winding+2*math.pi).abs()<.4
        achieved=torch.where(flip,stable&circle&(self.max_air_time>.12)&self.landed&arms&~self.early_air&(wheels>12.).all(-1),achieved)
        self.ref_hold=torch.where(achieved,self.ref_hold+self._env.step_dt,0.)
        self.ref_best_hold=torch.maximum(self.ref_best_hold,self.ref_hold)
        self.ref_success|=self.ref_hold>=1.5
        self.hold.copy_(self.ref_hold);self.best_hold.copy_(self.ref_best_hold);self.success.copy_(self.ref_success)


@configclass
class ReferenceCommandCfg(base.NativeSkillCommandCfg):
    class_type:type=ReferenceCommand
    reference_path:str=RELATIVE_REFERENCE
    sliding:bool=False
    boot_squat:bool=False


class ReferenceAction(base.NativeSkillAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.base_scale=torch.tensor([.20 if ('hip_' in n or 'knee_' in n) else .18 if ('shoulder_' in n or 'elbow_' in n) else .08 for n in Q1_INERTIA_POSITION_JOINTS],device=self.device)
        self.scale=self.base_scale

    def process_actions(self,actions):
        s=self._env.command_manager.get_term('skill');flip=s.mode==base.MODES.index('backflip')
        scale=self.base_scale.expand(self.num_envs,-1).clone()
        launch_or_flight=flip&((s.elapsed>=1.2)|s.has_air)
        scale=torch.where(launch_or_flight[:,None],.40,scale)
        # Keep learned flight swing, but enforce a bounded arms-down recovery.
        landed_arms=base.smooth((s.elapsed-s.touchdown_t)/.7)*s.landed.float()
        for i in s.arm_indices:scale[:,i]=scale[:,i]*(1-landed_arms)+.025*landed_arms
        self.scale=scale
        super().process_actions(actions.clamp(-2.,2.))
        r=self.robot.data;jac=self.robot.root_physx_view.get_jacobians()
        if jac.shape[1]!=self.robot.num_bodies:raise RuntimeError('Unexpected floating-base source Jacobian shape')
        offset=r.body_com_pos_w-r.body_pos_w
        jcom=jac[:,:,:3,6:]+torch.cross(jac[:,:,3:,6:].transpose(-2,-1),offset[:,:,None,:],dim=-1).transpose(-2,-1)
        mass=r.default_mass.to(self.device)
        gravity=9.81*(jcom[:,:,2,:]*mass[:,:,None]).sum(1)
        normal=s.sensor.data.net_forces_w[:,s.support_sensor_ids,2].clamp_min(0.)
        load=normal/normal.sum(-1,keepdim=True).clamp_min(1.)
        support=(jac[:,s.support_body_ids,2,6:]*load[:,:,None]).sum(1)*(9.81*mass.sum(1))[:,None]
        self.feedforward.copy_((gravity-support)[:,self.pids].clamp(-45.,45.))

    def apply_actions(self):
        s=self._env.command_manager.get_term('skill');r=self.robot.data
        self.robot.set_joint_position_target(self.target,joint_ids=self.pids)
        self.robot.set_joint_velocity_target(self.target_velocity,joint_ids=self.pids)
        self.robot.set_joint_effort_target(self.feedforward,joint_ids=self.pids)
        command=self._env.command_manager.get_command('base_velocity')
        mass=r.default_mass.to(self.device);com=(r.body_com_pos_w*mass[:,:,None]).sum(1)/mass.sum(1)[:,None]
        normal=s.sensor.data.net_forces_w[:,s.support_sensor_ids,2].clamp_min(0.)
        weight=normal/normal.sum(-1,keepdim=True).clamp_min(1.)
        cop=(r.body_pos_w[:,s.support_body_ids]*weight[:,:,None]).sum(1)
        foot=torch.where((normal.sum(-1)>1.)[:,None],cop,r.body_pos_w[:,self.feet].mean(1))
        offset=quat_apply_inverse(r.root_quat_w,com-foot)
        neutral=-torch.atan2(offset[:,0],offset[:,2]).clamp(-.42,.42)
        pitch=torch.atan2(r.projected_gravity_b[:,0],-r.projected_gravity_b[:,2])
        local=quat_apply_inverse(r.root_quat_w,r.root_pos_w-s.anchor)
        hold=(self.mode_is_stationary(s)).float()
        balance=65*(pitch-neutral)+12*r.root_ang_vel_b[:,1]+16*(r.root_lin_vel_b[:,0]-command[:,0])+2*local[:,0]*hold
        speed=balance+command[:,0]/.1
        physical=speed[:,None]+command[:,2:3]*torch.tensor([-.149,.149],device=self.device)/.1+4*self._raw[:,-2:]
        pushing=(s.mode==base.MODES.index('backflip'))&(s.elapsed>=1.2)&(s.elapsed<1.42)&~s.has_air
        physical=torch.where(pushing[:,None],20.*s.level+4*self._raw[:,-2:],physical)
        wheels=(physical*torch.tensor([-1.,1.],device=self.device)).clamp(-25.,25.)
        wheels=torch.where((s.has_air&~s.landed)[:,None],r.joint_vel[:,self.wids],wheels)
        for name,value in self.manual.items():
            if name in base.Q1_INERTIA_WHEEL_JOINTS:wheels[:,base.Q1_INERTIA_WHEEL_JOINTS.index(name)]=max(-25.,min(25.,value))
        self.robot.set_joint_velocity_target(wheels,joint_ids=self.wids)

    @staticmethod
    def mode_is_stationary(s):
        return (s.mode!=base.MODES.index('walk'))&(s.mode!=base.MODES.index('backflip'))


@configclass
class ReferenceActionCfg(base.NativeSkillActionCfg):
    class_type:type=ReferenceAction


def reset_reference(env,env_ids):
    s=env.command_manager.get_term('skill');robot=env.scene['robot']
    ids=robot._ALL_INDICES if env_ids is None else env_ids
    skill=env.cfg.reference_skill
    if env.cfg.boot_squat:
        phase=torch.zeros(len(ids),device=env.device)
        q23,height,_=s.trajectory.sample(phase)
        s.pending_reset_phase[ids]=0.
    elif skill in ('standup','kneel'):
        n=len(ids);level=float(s.level)
        phase=torch.rand(n,device=env.device)*level+(1-level)
        phase=torch.where((torch.rand(n,device=env.device)<.7)&(level>=.999),0.,phase)
        if not env.cfg.reference_phase_resets:phase.zero_()
        q23,height,_=s.trajectory.sample(phase,reverse=skill=='kneel')
        s.pending_reset_phase[ids]=phase
    else:
        endpoint=0. if env.cfg.boot_squat else 1.
        q23,height,_=s.trajectory.sample(torch.full((len(ids),),endpoint,device=env.device))
        s.pending_reset_phase[ids]=0.
    q=robot.data.default_joint_pos[ids].clone()
    for i,name in enumerate(Q1_INERTIA_JOINT_ORDER):q[:,robot.joint_names.index(name)]=q23[:,i]
    for follower,driver,multiplier in zip(Q1_INERTIA_MIMIC_JOINTS,Q1_INERTIA_MIMIC_DRIVERS,Q1_INERTIA_MIMIC_MULTIPLIERS):
        q[:,robot.joint_names.index(follower)]=q[:,robot.joint_names.index(driver)]*multiplier
    root=robot.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids]
    root[:,2]=env.scene.env_origins[ids,2]+height;root[:,7:]=0.
    phase_for_pitch=phase if skill in ('standup','kneel') else torch.full((len(ids),),endpoint,device=env.device)
    pitch=s.trajectory.sample_pitch(phase_for_pitch,reverse=skill=='kneel' and not env.cfg.boot_squat)
    root[:,3:7]=0.;root[:,3]=torch.cos(pitch/2);root[:,5]=torch.sin(pitch/2)
    robot.write_root_pose_to_sim(root[:,:7],env_ids=ids)
    robot.write_root_velocity_to_sim(root[:,7:],env_ids=ids)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)


def curriculum(env,env_ids,start_iter=0,steps_per_iteration=24):
    s=env.command_manager.get_term('skill');iteration=start_iter+env.common_step_counter//steps_per_iteration
    s.level=min(1.,.15+iteration/3000*.85)
    return s.level


def pose_tracking(env):
    s=base.skill(env);error=(s.robot.data.joint_pos[:,s.pids]-s.reference()).square().mean(-1)
    _,_,confidence=s.trajectory.sample(s.normalized_phase,reverse=env.cfg.reference_skill=='kneel')
    return torch.exp(-error/.06)*confidence.clamp(.2,1.)


def height_tracking(env):
    s=base.skill(env);r=s.robot.data;z=r.root_pos_w[:,2]-env.scene.env_origins[:,2]
    flight=(s.mode==base.MODES.index('backflip'))&(s.has_air|((s.elapsed>=1.2)&~s.landed))&~s.landed
    return torch.where(flight,0.,torch.exp(-(z-s.target_height()).square()/.012))


def terminal_hold(env):
    s=base.skill(env);return (s.hold/.5).clamp(0,3.)+8*s.success.float()


def contacts(env):
    s=base.skill(env);normal=s.sensor.data.net_forces_w[:,s.support_sensor_ids,2]
    flight=(s.mode==base.MODES.index('backflip'))&s.has_air&~s.landed
    return torch.where(flight,0.,((normal>12.).sum(-1)>=2).float())


def no_hop(env):
    s=base.skill(env);flip=s.mode==base.MODES.index('backflip')
    none=(s.sensor.data.net_forces_w.norm(dim=-1)<5).all(-1)
    return torch.where(flip,s.early_air.float(),none.float()+(s.robot.data.root_lin_vel_b[:,2].abs()-.35).clamp_min(0))


def landing_arms(env):
    s=base.skill(env);error=(s.robot.data.joint_pos[:,s.pids][:,s.arm_indices]-s.poses['stand'][s.arm_indices]).square().mean(-1)
    return torch.exp(-error/.03)*s.landed.float()


def push(env):return base.skill(env).push_reward
def recover(env):return base.skill(env).return_reward
def cycles(env):return base.skill(env).cycles.bonus


def failure(env):
    s=base.skill(env);r=s.robot.data;z=r.root_pos_w[:,2]-env.scene.env_origins[:,2]
    flip=s.mode==base.MODES.index('backflip')
    minimum=(s.target_height()-.28).clamp_min(.16)
    minimum=torch.where(flip,.07,minimum)
    tilt=(-r.projected_gravity_b[:,2]<.45)&~flip
    failed=(z<minimum)|tilt|s.illegal|s.early_air|((r.root_pos_w[:,:2]-env.scene.env_origins[:,:2]).norm(dim=-1)>12.)
    env._reference_last=dict(success=s.success.clone(),best_hold=s.best_hold.clone(),illegal=s.illegal.clone(),
        early_air=s.early_air.clone(),winding=s.winding.clone(),max_air=s.max_air_time.clone(),
        cycles=s.cycles.cycles.clone(),phase=s.normalized_phase.clone())
    return failed


def root_pitch_tracking(env):
    s=base.skill(env);r=s.robot.data
    target=s.trajectory.sample_pitch(s.normalized_phase,reverse=env.cfg.reference_skill=='kneel')
    moving=(s.mode==base.MODES.index('walk'))|(s.mode==base.MODES.index('backflip'))
    target=torch.where(moving,torch.zeros_like(target),target)
    pitch=torch.atan2(r.projected_gravity_b[:,0],-r.projected_gravity_b[:,2])
    flight=(s.mode==base.MODES.index('backflip'))&s.has_air&~s.landed
    return torch.where(flight,torch.zeros_like(pitch),torch.exp(-(pitch-target).square()/.04))
