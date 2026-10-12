"""Shared physical Q1 for web backflip, recovery, and kneel-to-stand policies.

Skill handovers change only controller state. Simulator pose writes are confined
to explicit Reset / Lie commands; Backflip and Unbox start from the current pose.
"""
import json,re,hashlib,math
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sensors import Imu,ImuCfg
from wheel_humanoid_lab.assets import IMU_CFG,Q1_WHEEL_CUBEMARS_CFG
from wheel_humanoid_lab.assets.wheel_humanoid import SKATE_HIP_PITCH,SKATE_KNEE
from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import _POS_ACTION_SCALE,_position_action_clip
from wheel_humanoid_lab.tasks.manager_based.posture.posture_env_cfg import POSTURE_POS_ACTION_SCALE
from arm_env import Env as ArmEnv, Config
from skills.policy import FrozenActor
from skills.recovery_runtime import RecoveryRuntime

RECOVERIES={
    'supine_elbow': 'q1_recovery_v3_supine_ppo.pt',
    'prone_elbow': 'q1_recovery_v3_prone_ppo.pt',
    'supine_gripper': 'q1_recovery_v3_supine_ppo_peak8000.pt',
    'prone_gripper': 'q1_recovery_v3_prone_gripper_ppo.pt',
    'boot_kneel': 'q1_unbox_transfer_ppo.pt',
    'landing_rise': 'q1_unbox_landing_ppo.pt',
}

class Env(ArmEnv):
    def _setup_scene(self):
        super()._setup_scene()
        self.imu=Imu(ImuCfg(prim_path='/World/envs/env_.*/Robot/'+IMU_CFG['parent_link'],
            offset=ImuCfg.OffsetCfg(pos=tuple(IMU_CFG['xyz'])),update_period=.005))
        self.scene.sensors['imu']=self.imu

    def __init__(self,cfg,root,checkpoint=None,teacher_path=None,**kwargs):
        self.mode='stand';self.ready=False;self.message='Standing; select a trained skill.'
        super().__init__(cfg,**kwargs)
        self.root=Path(root)
        self.recoveries={};self.recovery=None
        self.command=torch.zeros(self.num_envs,3,device=self.device)
        self.motor_wheels=torch.zeros(self.num_envs,2,device=self.device)
        self.last_skill_action=torch.zeros(self.num_envs,24,device=self.device)
        self.idle_target=self.home.expand(self.num_envs,-1).clone()
        self.manual_overrides={};self.position_bounds=self.limits.tolist()
        self.skate_default=self.robot.data.default_joint_pos[:,self.pids].clone()
        for i,name in enumerate(self.names):
            if name.endswith('_hip_pitch_joint'):self.skate_default[:,i]=SKATE_HIP_PITCH
            if name.endswith('_knee_joint'):self.skate_default[:,i]=SKATE_KNEE
        for name,file in RECOVERIES.items():
            if (self.root/'checkpoints'/file).is_file():
                self.recoveries[name]=RecoveryRuntime(self,self.root,name,file)
        self.regular={name:FrozenActor(self.root/'checkpoints'/file,self.device) for name,file in
            [('skate','q1_skate_ppo.pt'),('slide','q1_slide_ppo.pt'),('posture','q1_posture_ppo.pt')] if (self.root/'checkpoints'/file).is_file()}
        self.backflip_actor=None;self.backflip_checkpoint=None
        checkpoint=Path(checkpoint) if checkpoint else self.root/'checkpoints/backflip_arm_rest/model.pt'
        if checkpoint.is_file():
            proof={}
            metadata=checkpoint.parent/'runtime_metadata.json'
            if metadata.is_file():
                proof=json.loads(metadata.read_text())
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=proof['checkpoint_sha256']:
                    raise RuntimeError('Backflip checkpoint differs from validated model.')
                for name,expected in proof['files_sha256'].items():
                    if hashlib.sha256((self.root/name).read_bytes()).hexdigest()!=expected:
                        raise RuntimeError('Backflip plant/controller differs from validation: '+name)
            teacher_file=Path(teacher_path) if teacher_path else checkpoint.parent/'teacher.json'
            if teacher_path is None and proof.get('teacher_sha256') and hashlib.sha256(teacher_file.read_bytes()).hexdigest()!=proof['teacher_sha256']:
                raise RuntimeError('Backflip reference differs from validated model.')
            teacher=json.loads(teacher_file.read_text())
            self.set_parameters(torch.tensor(teacher['parameters'],device=self.device).expand(self.num_envs,-1))
            self.backflip_actor=FrozenActor(checkpoint,self.device);self.backflip_checkpoint=str(checkpoint)
        self.pose_timer=0.;self.prepare_hold=0.;self.posture_target=0.;self.kneel_ready_hold=0.
        self.hold_phase=torch.zeros(self.num_envs,device=self.device)
        self.stand_anchor=self.robot.data.root_pos_w.clone();self.ready=True

    def _state(self):
        if getattr(self,'mode','stand')=='backflip':super()._state()

    def _phase(self):
        t=self.episode_length_buf.float().clamp(max=299)*self.step_dt
        if hasattr(self,'hold_phase'):t=torch.where(self.hold_phase>0,self.hold_phase,t)
        return t

    def _get_rewards(self):return torch.zeros(self.num_envs,device=self.device)
    def _get_dones(self):
        # Interactive skills must keep their landed state instead of auto-respawn.
        z=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        return z,z

    def _get_observations(self):
        if getattr(self,'mode','stand')=='backflip':return super()._get_observations()
        return {'policy':torch.zeros(self.num_envs,83,device=self.device)}

    def _clear_command_queues(self):
        # Flush queued commands when switching controllers. Preserve motor
        # voltage, friction, configured latency and all physical robot states.
        ids=self.robot._ALL_INDICES
        for actuator in self.robot.actuators.values():
            for name in ['_pos_buf','_vel_buf','_eff_buf']:
                buffer=getattr(actuator,name,None)
                if buffer is not None:buffer.reset(ids)

    def _clear_flip(self):
        self._clear_command_queues()
        self.hold_phase.zero_()
        for name in ['winding','air','air_max','hold','peak','takeoff_t','touchdown_t','max_progress','progress_delta','last_progress','takeoff_omega','best_hold','takeoff_vz','takeoff_momentum']:
            getattr(self,name).zero_()
        for name in ['jumped','failed','success','lifted','has_air','prehop','landed','failed_before_air']:
            getattr(self,name).fill_(False)
        self.prev_quat.copy_(self.robot.data.root_quat_w)
        self.actions.zero_();self.previous.zero_();self.prev_target.copy_(self.idle_target)
        self.prev_prior.copy_(self.home);self.target_velocity.zero_();self.tick=-1
        self.episode_length_buf.zero_();self.updates=20000

    def _standing_wheels(self):
        r=self.robot.data
        com=(r.body_com_pos_w*self.mass[:,:,None]).sum(1)/self.mass.sum(1)[:,None]
        ids=self.robot.find_bodies(['l_wheel_link','r_wheel_link'],preserve_order=True)[0]
        support=r.body_pos_w[:,ids].mean(1)
        offset=quat_apply_inverse(r.root_quat_w,com-support)
        neutral=-torch.atan2(offset[:,0],offset[:,2]).clamp(-.15,.15)
        pitch=torch.atan2(r.projected_gravity_b[:,0],-r.projected_gravity_b[:,2])
        error=quat_apply_inverse(r.root_quat_w,r.root_pos_w-self.stand_anchor)[:,0]
        velocity=65*(pitch-neutral)+12*r.root_ang_vel_b[:,1]+14*r.root_lin_vel_b[:,0]+3*error
        return velocity[:,None].expand(-1,2).clamp(-20,20)

    def _standing_feedforward(self):
        r=self.robot.data;jac=self.robot.root_physx_view.get_jacobians()
        if jac.shape[1]!=self.robot.num_bodies:return self.ff
        offset=r.body_com_pos_w-r.body_pos_w
        linear=jac[:,:,:3,6:];angular=jac[:,:,3:,6:]
        jcom=linear+torch.cross(angular.transpose(-2,-1),offset[:,:,None,:],dim=-1).transpose(-2,-1)
        gravity=9.81*(jcom[:,:,2,:]*self.mass[:,:,None]).sum(1)
        feet=self.robot.find_bodies(['l_wheel_link','r_wheel_link'],preserve_order=True)[0]
        # Floating-base Jacobians account for hanging leg mass. Standing also
        # needs the joint torque opposing the two feet's weight-bearing forces.
        support=jac[:,feet,2,6:].sum(1)*(9.81*self.mass.sum(1)/2)[:,None]
        return (gravity-support)[:,self.pids].clamp(-45,45)

    def request(self,name):
        if name=='reset':
            self.mode='stand';super()._reset_idx(self.robot._ALL_INDICES)
            self.idle_target.copy_(self.home);self.pose_timer=0.;self.motor_wheels.zero_()
            self.stand_anchor.copy_(self.robot.data.root_pos_w)
            self.message='Reset to slightly bent standing pose.';return True
        if name=='backflip':
            if self.backflip_actor is None:
                self.message='Backflip model unavailable.';return False
            upright=-self.robot.data.projected_gravity_b[:,2]
            height=self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]
            if not bool(((upright>.94)&(height>.72)&(self.robot.data.root_lin_vel_w[:,:2].norm(dim=-1)<.8)).all()):
                self.message='Backflip requires stable standing. Use Unbox from kneel first.';return False
            self.mode='prepare_backflip';self.pose_timer=0.;self.prepare_hold=0.
            self.prepare_start=self.robot.data.joint_pos[:,self.pids].clone()
            self.command.zero_()
            if bool(self._launch_ready().all()):
                self.idle_target.copy_(self.home);self._clear_flip();self.mode='backflip'
                self.message='Backflip PPO running from the current slight knee bend.'
            else:self.message='Moving smoothly to the trained slight knee bend.'
            return True
        if name in ['unbox','getup']:
            if self.mode=='kneeling':
                self.message='Wait for the kneeling motion to settle before Unbox.';return False
            if self.mode=='backflip' and not bool(self.success.all()):
                self.message='Wait for Backflip to finish the arms-down landing hold.';return False
            height=self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]
            upright=-self.robot.data.projected_gravity_b[:,2]
            if not bool(((height>.24)&(height<.68)&(upright>.85)).all()):
                self.message='Unbox requires an upright kneel. Select Kneel or finish Backflip.';return False
            knees=self.robot.data.joint_pos[:,self.kids].mean(-1)
            hips=self.robot.find_joints(['l_hip_pitch_joint','r_hip_pitch_joint'],preserve_order=True)[0]
            folded=(knees>2.48)&(self.robot.data.joint_pos[:,hips].mean(-1)>-1.15)
            landing=(self.mode=='backflip' or (self.mode not in ['kneel','kneeling'] and bool(folded.all()))) and 'landing_rise' in self.recoveries
            self._clear_command_queues()
            self.recovery=self.recoveries['landing_rise' if landing else 'boot_kneel']
            if landing:self.recovery.start_landing()
            else:self.recovery.start_kneel()
            self.mode='unbox';self.command.zero_();self.message='Unbox: trained kneel-to-stand PPO from current pose.';return True
        if name in ['supine','prone']:
            self.recovery=self.recoveries[name+'_gripper'];self.recovery.place_lie(0 if name=='supine' else 1)
            self.idle_target.copy_(self.robot.data.joint_pos[:,self.pids]);self.motor_wheels.zero_()
            self.mode=name;self.message='Floor pose ready; select Recovery Elbow or Gripper.';return True
        if name.startswith('recovery_') and name.removeprefix('recovery_') in RECOVERIES:
            variant=name.removeprefix('recovery_');orientation=variant.split('_')[0]
            selected=self.recoveries[variant]
            if self.mode not in ['supine','prone']:
                self.message='Select the matching Supine / Prone floor pose first.';return False
            if self.mode!=orientation:
                self.message='Select the matching floor orientation first.';return False
            self.recovery=selected
            # Initialize controller timers/targets for the current floor pose;
            # use the explicit lie command to place the simulation, never here.
            s=selected.state;s.mode.fill_(0 if orientation=='supine' else 1)
            s.stage.zero_();s.elapsed.zero_();s.hold.zero_();s.kneel_hold.zero_();s.stand_hold.zero_()
            s.knelt.fill_(False);s.stood.fill_(False);s.planted.fill_(False)
            if hasattr(s,'boot_assist'):s.boot_assist.fill_(False)
            s.upright_active.fill_(False);s.upright_captured.fill_(False);s.upright_knelt.fill_(False)
            s.upright_age.zero_();s.upright_hold.zero_();s.upright_command.zero_()
            s.held_command.zero_()
            s.transfer_held.fill_(False);s.filtered_force.zero_();s.last_step=-1;s.dirty=True
            s.start_q.copy_(self.robot.data.joint_pos);s.command.copy_(s.start_q)
            selected._reset_observations();selected.start_floor()
            self.mode=name;self.message='Running '+variant+' PPO with its matching controller.';return True
        if name in ['recovery','recovery_ppo']:
            orientation='prone' if self.mode=='prone' else 'supine'
            return self.request('recovery_'+orientation+'_gripper')
        if name=='stand' and (self.mode in ['kneel','kneeling','backflip_hold'] or (self.mode=='backflip' and bool(self.success.all()))):
            return self.request('unbox')
        if name=='kneel':
            self.mode='kneeling';self.kneel_ready_hold=0.;self.posture_target=1.;self.message='Trained posture PPO: kneel.';return True
        if name in ['skate','slide']:
            height=self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]
            if not bool((height>.72).all()):
                self.message='Use Unbox to stand before skate / slide.';return False
            self.recoveries['boot_kneel'].last_action.zero_()
            self.mode=name;self.message=name+' PPO active.';return True
        self.message='Unknown or unavailable skill: '+name;return False

    def _launch_ready(self):
        r=self.robot.data
        return ((r.joint_pos[:,self.pids]-self.home).abs().amax(-1)<.15)&(r.root_lin_vel_w[:,:2].norm(dim=-1)<.05)&(r.root_ang_vel_b.norm(dim=-1)<.15)&(r.projected_gravity_b[:,0].abs()<.06)

    def actor_action(self):
        if self.mode=='backflip':return self.backflip_actor(self._get_observations()['policy'])
        return torch.zeros(self.num_envs,24,device=self.device)

    def _pre_physics_step(self,actions):
        if not self.ready:
            self.target=self.home.expand(self.num_envs,-1).clone();return
        self.pose_timer+=self.step_dt
        if self.mode=='backflip':
            super()._pre_physics_step(actions);return
        if self.mode=='prepare_backflip':
            x=min(1.,self.pose_timer/1.2);x=x*x*(3-2*x)
            self.idle_target=self.prepare_start+x*(self.home-self.prepare_start)
            settled=self._launch_ready()
            self.prepare_hold=self.prepare_hold+self.step_dt if x==1. and bool(settled.all()) else 0.
            if self.prepare_hold>=.04:
                self._clear_flip();self.mode='backflip'
                self.message='Backflip PPO running.'
            elif self.pose_timer>8.:
                self.mode='stand';self.message='Standing preparation did not settle; Reset before retry.'
            # Preparation must keep closed-loop wheel balance running.
            self.motor_wheels=self._standing_wheels()
        elif self.mode in ['unbox','unbox_hold'] or self.mode.startswith('recovery_'):
            self.idle_target,self.motor_wheels=self.recovery.act()
        elif self.mode in ['kneeling','kneel','skate','slide']:
            name='posture' if self.mode in ['kneeling','kneel'] else self.mode
            runtime=self.recoveries['boot_kneel']
            # Reuse the 78-D proprio contract and append the skill command block.
            proprio=runtime.observation()[:,:78]
            cmd=torch.zeros(self.num_envs,14,device=self.device);cmd[:,:3]=self.command
            cmd[:,-1]=1.;cmd[:,7]=self.posture_target if name=='posture' else 0.
            action=self.regular[name](torch.cat((proprio,cmd),dim=-1))
            scale_spec=POSTURE_POS_ACTION_SCALE if name=='posture' else _POS_ACTION_SCALE
            clip=_position_action_clip(scale_spec)
            target=self.skate_default.clone()
            for i,joint in enumerate(self.names):
                scale=next(v for p,v in scale_spec.items() if re.fullmatch(p,joint))
                target[:,i]=(target[:,i]+scale*action[:,i]).clamp(*clip[joint])
            self.idle_target=target;self.motor_wheels=(8*action[:,22:]).clamp(-20,20)
            runtime.last_action.copy_(action)
        elif self.mode in ['supine','prone']:
            self.motor_wheels.zero_()
        else:
            self.motor_wheels=self._standing_wheels()
            if self.mode=='stand':
                self.idle_target=self.home.expand(self.num_envs,-1).clone()
                for name,value in self.manual_overrides.items():
                    if not math.isfinite(value):continue
                    if name in self.names:
                        i=self.names.index(name);lo,hi=self.position_bounds[i]
                        self.idle_target[:,i]=min(hi,max(lo,value))
                    elif name in ['l_wheel_joint','r_wheel_joint']:
                        self.motor_wheels[:,0 if name.startswith('l_') else 1]=min(25.,max(-25.,value))
        self.target=self.idle_target

    def _apply_action(self):
        if self.mode=='backflip':super()._apply_action();return
        if not self.ff_ready:self._gravity_feedforward()
        recovery_mode=self.mode in ['unbox','unbox_hold'] or self.mode.startswith('recovery_')
        self.robot.set_joint_position_target(self.target,joint_ids=self.pids)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.target),joint_ids=self.pids)
        regular_mode=self.mode in ['kneeling','kneel','skate','slide','supine','prone']
        self.robot.set_joint_effort_target(torch.zeros_like(self.target) if recovery_mode or regular_mode else self._standing_feedforward(),joint_ids=self.pids)
        self.robot.set_joint_velocity_target(self.motor_wheels,joint_ids=self.wids)
        self.robot.set_joint_effort_target(torch.zeros_like(self.motor_wheels),joint_ids=self.wids)
        self.robot.set_joint_position_target(.444444*self.robot.data.joint_pos[:,self.kids],joint_ids=self.rids)
        self.robot.set_joint_velocity_target(.444444*self.robot.data.joint_vel[:,self.kids],joint_ids=self.rids)

    def after_step(self):
        if self.mode=='backflip':
            self._state()
            if getattr(self,'batch_trials',False):
                self.hold_phase=torch.where(self.success&(self.hold_phase==0),self._phase(),self.hold_phase)
                return
            if bool(self.failed.any()|self.prehop.any()):
                self.mode='failed';self.idle_target=self.target.clone();self.message='Backflip failed; Reset to retry.'
            elif bool(self.success.all()):
                self.hold_phase=torch.where(self.hold_phase==0,self._phase(),self.hold_phase)
                # Keep the same trained landing controller active after success.
                self.message='Backflip complete: arms down, balanced kneel. Press Unbox to stand.'
        if self.mode in ['unbox','unbox_hold'] or self.mode.startswith('recovery_'):
            s=self.recovery.state;s.update()
            if self.mode in ['unbox','unbox_hold'] and bool(((self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]<.18)&(s.elapsed>.7)).any()):
                self.mode='failed';self.idle_target=self.target.clone();self.message='Unbox lost balance; Reset to retry.'
                return
            if bool((s.stand_hold>=1.).all()):
                if self.mode in ['unbox','unbox_hold']:
                    self.mode='unbox_hold';self.message='Standing stable; the new Unbox PPO keeps balancing.'
                else:
                    self.recoveries['boot_kneel'].last_action.zero_()
                    self.mode='skate';self.message='Standing stable; skate PPO balances.'
        if self.mode=='kneeling':
            z=self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]
            ready=(z>.24)&(z<.60)&(-self.robot.data.projected_gravity_b[:,2]>.85)&(self.robot.data.root_lin_vel_w.norm(dim=-1)<.25)&(self.robot.data.root_ang_vel_b.norm(dim=-1)<.6)
            self.kneel_ready_hold=self.kneel_ready_hold+self.step_dt if bool(ready.all()) else 0.
            if self.kneel_ready_hold>=.3:
                self.mode='kneel';self.message='Stable kneel; press Unbox to stand.'

    def snapshot(self):
        r=self.robot.data;q=r.root_quat_w[0];pos=r.root_pos_w[0]-self.scene.env_origins[0]
        rows=torch.stack((r.joint_pos[0],r.joint_vel[0],r.applied_torque[0]),dim=-1).tolist()
        telemetry=torch.cat((q,pos,self.success[:1].float(),self.arm_status()[0][:1],self.best_hold[:1],self.prehop[:1].float(),r.root_lin_vel_w[0],r.root_ang_vel_b[0])).tolist()
        actors={k:v.actor for k,v in self.recoveries.items()};actors.update(self.regular)
        if self.backflip_actor is not None:actors['backflip']=self.backflip_actor
        return {'ok':True,'pose':self.mode,'message':self.message,
            'manual_control_available':self.mode=='stand',
            'cmd':{'vx':float(self.command[0,0]),'vy':0.,'yaw':float(self.command[0,2])},
            'base':dict(zip(['qw','qx','qy','qz'],telemetry[:4]),z=telemetry[6],x=telemetry[4],y=telemetry[5],lin_vel_w=telemetry[11:14],ang_vel_b=telemetry[14:17]),
            'joints':{name:dict(zip(['pos','vel','torque'],row)) for name,row in zip(self.robot.joint_names,rows)},
            'policies':{'backflip':self.backflip_actor is not None,'unbox':True,'recovery':True,
                'recovery_supine':True,'recovery_prone':True,**{k:True for k in RECOVERIES}},
            'models':{k:{'checkpoint':getattr(a,'checkpoint_path',None),'sha256':getattr(a,'checkpoint_sha256',None),'iteration':getattr(a,'training_iteration',None)} for k,a in actors.items()},
            'backflip':{'success':bool(telemetry[7]),'arms_error_rad':telemetry[8],
                'hold_s':telemetry[9],'early_hop':bool(telemetry[10])}}
