"""Single load/push backflip with event-gated flight and one-way progress rewards."""
import math,json
from pathlib import Path
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul,quat_conjugate,quat_apply,quat_apply_inverse
from q1_backflip.env import Env as BaseEnv,Config as BaseConfig

# [crouch duration, knee, hip offset, waist, push duration, hip, knee,
#  load shoulder, push shoulder, push waist, tuck knee, tuck hip,
#  tuck waist, tuck delay, balance gain, landing kneel blend]
DEFAULT=[.38,1.4,-.12,-.15,.14,.35,-.10,-1.3,1.8,.20,2.3,.80,.25,.04,20.,1.,12.,.03,-.25]
LOW=[.20,.9,-.35,-.45,.07,-.50,-.14,-2.1,-.5,-.5,1.6,-1.75,-.4,.0,8.,0.,-8.,-.02,-.6]
HIGH=[.65,2.0,.10,.35,.25,1.10,.30,.8,2.2,.65,2.5,1.1,.9,.14,40.,1.,20.,.14,.1]

@configclass
class Config(BaseConfig):
    episode_length_s=6.0
    landing='kneel'

class Env(BaseEnv):
    def _make_reference(self):
        return self.home.repeat(301,1),{'source':'Q1 constrained load/push trajectory','takeoff':.78,'touchdown':1.55,'landing':self.cfg.landing,'fully_dynamic':True,'self_collisions':False}

    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        n=self.num_envs;dev=self.device
        self.params=torch.tensor(DEFAULT,device=dev).expand(n,-1).clone()
        self.load_start=.40
        self.has_air=torch.zeros(n,dtype=torch.bool,device=dev)
        self.takeoff_t=torch.zeros(n,device=dev);self.touchdown_t=self.takeoff_t.clone()
        self.max_progress=self.takeoff_t.clone();self.progress_delta=self.takeoff_t.clone()
        self.last_progress=self.takeoff_t.clone();self.prehop=self.has_air.clone();self.landed=self.has_air.clone()
        self.takeoff_omega=self.takeoff_t.clone();self.best_hold=self.takeoff_t.clone()
        self.failed_before_air=self.has_air.clone()
        self.best_push_height=self.takeoff_t.clone()
        self.prev_target=self.home.expand(n,-1).clone();self.prev_prior=self.prev_target.clone();self.target_velocity=torch.zeros_like(self.prev_target)
        self.pose_history=[]
        self.search_mode=False
        self.ff=torch.zeros(n,22,device=dev)
        # Static gravity feed-forward under the same current/torque limits; initialized once.
        self.ff_ready=False
        allowed=set(self.wb+self.rb)
        self.bad=[i for i in range(len(self.contacts.body_names)) if i not in allowed]
        self.freeze_search=False
        self.mass=self.robot.data.default_mass.to(dev)
        self.initial_com_z=(self.robot.data.body_com_pos_w[:,:,2]*self.mass).sum(dim=1)/self.mass.sum(dim=1)-self.scene.env_origins[:,2]
        self.peak_com=self.initial_com_z.clone();self.takeoff_vz=torch.zeros(n,device=dev)
        self.inertia=self.robot.data.default_inertia.to(dev).reshape(n,-1,3,3)
        self.takeoff_momentum=torch.zeros(n,device=dev)

    def set_parameters(self,p):
        self.params.copy_(p)
        self.motion_meta['parameters']=self.params[0].tolist()

    def _commands(self,t):
        p=self.params
        stand=self.home.expand(self.num_envs,-1).clone()
        crouch=stand.clone();push=stand.clone();tuck=stand.clone();land=stand.clone()
        for side in ['l','r']:
            i=lambda suffix:self.names.index(f'{side}_{suffix}_joint')
            crouch[:,i('knee')]=p[:,1];crouch[:,i('hip_pitch')]=-.5*p[:,1]+p[:,2]
            crouch[:,i('shoulder_pitch')]=p[:,7];crouch[:,i('elbow')]=-1.1
            push[:,i('knee')]=p[:,6];push[:,i('hip_pitch')]=p[:,5]
            push[:,i('shoulder_pitch')]=p[:,8];push[:,i('elbow')]=-.3
            tuck[:,i('knee')]=p[:,10];tuck[:,i('hip_pitch')]=p[:,11]
            tuck[:,i('shoulder_pitch')]=p[:,8];tuck[:,i('elbow')]=-1.3
            kneel=(p[:,15]>.5).float()
            land[:,i('knee')]=.3*(1-kneel)+2.24*kneel
            land[:,i('hip_pitch')]=-.15*(1-kneel)-1.02*kneel
            land[:,i('shoulder_pitch')]=.35;land[:,i('elbow')]=-1.1
        wi=self.names.index('waist_pitch_joint')
        crouch[:,wi]=p[:,3];push[:,wi]=p[:,9];tuck[:,wi]=p[:,12];land[:,wi]=.15
        push_start=self.load_start+p[:,0]
        # Quintic load avoids a sharp jerk; fast but finite push establishes angular momentum.
        s=((t-self.load_start)/p[:,0]).clamp(0,1);smooth=s*s*s*(10-15*s+6*s*s)
        ref=stand+(crouch-stand)*smooth[:,None]
        s=((t-push_start)/p[:,4]).clamp(0,1);s=s*s*(3-2*s)
        ref=torch.where((t>=push_start)[:,None],crouch+(push-crouch)*s[:,None],ref)
        hs=((t-push_start-p[:,17])/p[:,4]).clamp(0,1);hs=hs*hs*(3-2*hs)
        for side in ['l','r']:
            i=self.names.index(f'{side}_hip_pitch_joint')
            ref[:,i]=torch.where(t>=push_start,crouch[:,i]+(push[:,i]-crouch[:,i])*hs,ref[:,i])
        # Tuck ONLY after actual rising takeoff; cannot turn the flight reference into a second squat.
        flight_t=(t-self.takeoff_t-p[:,13]).clamp_min(0)
        s=(flight_t/.12).clamp(0,1);s=s*s*(3-2*s)
        ref=torch.where(self.has_air[:,None],push+(tuck-push)*s[:,None],ref)
        # Expand before finishing the circle, while descending; landing remains contact gated.
        extend=(self.max_progress>5.0)&(self.robot.data.root_lin_vel_w[:,2]<.8)
        ref=torch.where((extend|self.landed)[:,None],land,ref)
        return ref.clamp(self.limits[:,0]+.02,self.limits[:,1]-.02)

    def _target(self):return self._commands(self._phase())

    def _pre_physics_step(self,actions):
        if not torch.isfinite(actions).all():raise FloatingPointError('nonfinite actions')
        self.previous.copy_(self.actions);self.actions.copy_(actions.clamp(-2,2))
        t=self._phase();push=self.load_start+self.params[:,0]
        prior=self._commands(t)
        # No learned launch impulse during the initial hold; bounded residual during a single load.
        scale=torch.where(t<self.load_start,0.,torch.where(t<push,.12,.50))
        target=(prior+scale[:,None]*self.actions[:,:22]).clamp(self.limits[:,0],self.limits[:,1])
        velocity_limit=self.robot.data.joint_vel_limits[:,self.pids]*.8
        change=(target-self.prev_target).clamp(-velocity_limit*self.step_dt,velocity_limit*self.step_dt)
        for i,name in enumerate(self.names):
            if any(x in name for x in ['wrist','gripper','neck','waist_yaw','waist_roll']):
                change[:,i]=self.home[i]-self.prev_target[:,i]
        self.target=self.prev_target+change
        prior_change=(prior-self.prev_prior).clamp(-velocity_limit*self.step_dt,velocity_limit*self.step_dt)
        self.target_velocity=prior_change/self.step_dt
        self.prev_prior.add_(prior_change)
        self.prev_target.copy_(self.target)

    def _gravity_feedforward(self):
        # Joint-side gravity torque at reset pose from the articulation Jacobian.
        jac=self.robot.root_physx_view.get_jacobians()
        if jac.shape[1]!=self.robot.num_bodies:return
        mass=self.robot.data.default_mass.to(self.device)
        offset=self.robot.data.body_com_pos_w-self.robot.data.body_pos_w
        linear=jac[:,:,:3,6:]
        angular=jac[:,:,3:,6:]
        jcom=linear+torch.cross(angular.transpose(-2,-1),offset[:,:,None,:],dim=-1).transpose(-2,-1)
        tau=9.81*(jcom[:,:,2,:]*mass[:,:,None]).sum(dim=1)
        self.ff.copy_(tau[:,self.pids].clamp(-45,45));self.ff_ready=True
        print('GRAVITY_FF',dict(zip(self.names,self.ff[0].tolist())),flush=True)

    def _apply_action(self):
        if not self.ff_ready:self._gravity_feedforward()
        self.robot.set_joint_position_target(self.target,joint_ids=self.pids)
        self.robot.set_joint_velocity_target(self.target_velocity,joint_ids=self.pids)
        t=self._phase();ground=~self.has_air|self.landed
        ff_gain=torch.where(t<self.load_start,1.,.5)*ground
        self.robot.set_joint_effort_target(self.ff*ff_gain[:,None],joint_ids=self.pids)
        pitch=torch.atan2(self.robot.data.projected_gravity_b[:,0],-self.robot.data.projected_gravity_b[:,2])
        load=((t-self.load_start)/self.params[:,0]).clamp(0,1)
        desired=torch.where((t>=self.load_start)&(~self.has_air),self.params[:,18]*load,0.)
        velocity=self.params[:,14]*(pitch-desired)+3.*self.robot.data.root_ang_vel_b[:,1]+2.*self.robot.data.root_lin_vel_b[:,0]
        wheels=(velocity[:,None]+4*self.actions[:,22:].clamp(-1,1)).clamp(-20,20)
        wheels=torch.where((t<self.load_start)[:,None],velocity[:,None].expand(-1,2).clamp(-12,12),wheels)
        push_start=self.load_start+self.params[:,0]
        wheels=torch.where(((t>=push_start)&(~self.has_air)&(~self.landed))[:,None],(self.params[:,16,None]+4*self.actions[:,22:].mean(dim=1,keepdim=True).clamp(-1,1)).expand(-1,2).clamp(-20,20),wheels)
        wheels=torch.where((self.has_air&(~self.landed))[:,None],torch.zeros_like(wheels),wheels)
        air_coast=self.has_air&(~self.landed)
        wheels=torch.where(air_coast[:,None],self.robot.data.joint_vel[:,self.wids],wheels)
        self.robot.set_joint_velocity_target(wheels,joint_ids=self.wids)
        wheel_ff=torch.where(((t>=push_start)&(~self.has_air)&(~self.landed))[:,None],(2*self.params[:,16,None]+8*self.actions[:,22:].mean(dim=1,keepdim=True).clamp(-1,1)).expand(-1,2).clamp(-40,40),torch.zeros_like(wheels))
        self.robot.set_joint_effort_target(wheel_ff,joint_ids=self.wids)
        self.robot.set_joint_position_target(.444444*self.robot.data.joint_pos[:,self.kids],joint_ids=self.rids)
        self.robot.set_joint_velocity_target(.444444*self.robot.data.joint_vel[:,self.kids],joint_ids=self.rids)

    def _state(self):
        if self.tick==self.common_step_counter:return
        self.tick=self.common_step_counter
        dq=quat_mul(self.robot.data.root_quat_w,quat_conjugate(self.prev_quat))
        dq=torch.where(dq[:,:1]<0,-dq,dq);norm=dq[:,1:].norm(dim=-1)
        self.delta=2*torch.atan2(norm,dq[:,0])*dq[:,2]/norm.clamp_min(1.e-8)
        self.winding+=self.delta;self.prev_quat.copy_(self.robot.data.root_quat_w)
        forces=self.contacts.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        self.wcontact=forces[:,self.wb]>5;self.rcontact=forces[:,self.rb]>5
        self.failed|=(forces[:,self.bad]>10).any(dim=1)
        airborne=~(forces>5).any(dim=1)
        self.air=torch.where(airborne,self.air+self.step_dt,torch.zeros_like(self.air))
        self.air_max=torch.maximum(self.air_max,self.air)
        z=self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2];self.peak=torch.maximum(self.peak,z)
        if not hasattr(self,'params'):return
        t=self._phase();p=self.params;push=self.load_start+p[:,0]
        all_air=airborne
        com_z=(self.robot.data.body_com_pos_w[:,:,2]*self.mass).sum(dim=1)/self.mass.sum(dim=1)-self.scene.env_origins[:,2]
        com_vz=(self.robot.data.body_com_lin_vel_w[:,:,2]*self.mass).sum(dim=1)/self.mass.sum(dim=1)
        self.peak_com=torch.maximum(self.peak_com,com_z)
        genuine=all_air&(com_vz>.4)&(t>push)&(~self.failed)
        new=genuine&(~self.has_air)&(~self.landed)
        self.takeoff_vz=torch.where(new,com_vz,self.takeoff_vz)
        q=self.robot.data.body_com_quat_w;omega=self.robot.data.body_com_ang_vel_w
        local=quat_apply_inverse(q,omega)
        spin=quat_apply(q,(self.inertia@local.unsqueeze(-1)).squeeze(-1)).sum(dim=1)
        com=(self.robot.data.body_com_pos_w*self.mass[:,:,None]).sum(dim=1)/self.mass.sum(dim=1)[:,None]
        vel=self.robot.data.body_com_lin_vel_w;vcom=(vel*self.mass[:,:,None]).sum(dim=1)/self.mass.sum(dim=1)[:,None]
        orbit=(torch.cross(self.robot.data.body_com_pos_w-com[:,None,:],vel-vcom[:,None,:],dim=-1)*self.mass[:,:,None]).sum(dim=1)
        momentum=-(spin+orbit)[:,1]
        self.takeoff_momentum=torch.where(new,momentum,self.takeoff_momentum)
        self.takeoff_t=torch.where(new,t,self.takeoff_t);self.takeoff_omega=torch.where(new,-self.robot.data.root_ang_vel_b[:,1],self.takeoff_omega)
        self.has_air|=genuine
        self.jumped|=self.has_air&(self.air>=.04);self.lifted|=self.has_air
        # Early hop is a task failure, even if it happens to give height/rotation rewards.
        early=all_air&(self.air>.02)&(t<push-.02)&(self.robot.data.root_lin_vel_w[:,2]>.12)
        self.prehop|=early
        self.failed_before_air|=self.failed&(~self.has_air)
        touch=self.has_air&(t-self.takeoff_t>.12)&(self.wcontact.any(dim=1)|self.rcontact.any(dim=1))
        self.touchdown_t=torch.where(touch&(~self.landed),t,self.touchdown_t);self.landed|=touch
        active=self.has_air&(~self.landed)&(~self.failed)
        progress=(-self.winding).clamp(0,2*math.pi+.5)
        self.last_progress.copy_(self.max_progress)
        self.max_progress=torch.where(active,torch.maximum(self.max_progress,progress),self.max_progress)
        self.progress_delta=self.max_progress-self.last_progress
        upright=-self.robot.data.projected_gravity_b[:,2]
        kneel=p[:,15]>.5
        support=torch.where(kneel,self.wcontact.any(dim=1)&self.rcontact.any(dim=1)&((self.wcontact.sum(dim=1)+self.rcontact.sum(dim=1))>=3),self.wcontact.all(dim=1))
        stable=(upright>.92)&(self.robot.data.root_ang_vel_b.norm(dim=1)<1.)&(self.robot.data.root_lin_vel_w[:,:2].norm(dim=1)<.35)
        full=(self.winding+2*math.pi).abs()<.30
        ok=self.landed&full&(self.max_progress>5.9)&support&stable&(~self.failed)&(~self.prehop)
        self.hold=torch.where(ok,self.hold+self.step_dt,torch.zeros_like(self.hold));self.best_hold=torch.maximum(self.best_hold,self.hold)
        self.success|=self.hold>=2.

    def _get_rewards(self):
        self._state();t=self._phase();r=self.robot.data
        prep=t<self.load_start+self.params[:,0]
        tracking=torch.exp(-((r.joint_pos[:,self.pids]-self._commands(t))**2).mean(dim=-1)/.12)
        one_way=25*self.progress_delta*(self.takeoff_vz/3.).clamp(0,1).square()
        forward=(self.delta.clamp_min(0)*self.has_air)*15
        # No standalone height reward: jump without a full turn cannot keep earning.
        reward=one_way-forward+(2*tracking*prep+2*(self.hold>0).float()-1.-.015*(self.actions-self.previous).square().sum(dim=-1)-.002*self.actions.square().sum(dim=-1)-1.e-5*r.applied_torque.square().sum(dim=-1)-1.*r.root_ang_vel_b[:,[0,2]].square().sum(dim=-1))*self.step_dt
        reward-=30*self.reset_terminated.float()+50*self.prehop.float()*self.step_dt
        finish=self.max_progress>5.0
        reward+=finish.float()*(12*torch.exp(-((self.winding+2*math.pi)/.35).square())-.1*r.root_ang_vel_b[:,1].square())*self.step_dt
        reward+=200*(self.hold>=2.).float()*self.step_dt
        if not torch.isfinite(reward).all():raise FloatingPointError('nonfinite reward')
        return reward

    def _get_dones(self):
        self._state()
        failed,time_out=super()._get_dones()
        return failed|self.prehop,time_out

    def _reset_idx(self,ids):
        if ids is None:ids=self.robot._ALL_INDICES
        if hasattr(self,'params'):
            extra={'max_backturn':float(self.max_progress[ids].mean()),'early_hop':float(self.prehop[ids].float().mean()),'takeoff_omega':float(self.takeoff_omega[ids].mean()),'max_hold':float(self.best_hold[ids].mean())}
        super()._reset_idx(ids)
        if hasattr(self,'params'):
            self.extras['log'].update({f'Backflip/{k}':v for k,v in extra.items()})
            self.metric_history[-1].update(extra)
            for b in [self.takeoff_t,self.touchdown_t,self.max_progress,self.progress_delta,self.last_progress,self.takeoff_omega,self.best_hold]:b[ids]=0
            for b in [self.has_air,self.prehop,self.landed,self.failed_before_air]:b[ids]=False
            self.prev_target[ids]=self.home;self.prev_prior[ids]=self.home;self.target_velocity[ids]=0
            self.peak_com[ids]=self.initial_com_z[ids];self.takeoff_vz[ids]=0;self.takeoff_momentum[ids]=0
