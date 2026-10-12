"""Independent Q1 backflip: CubeMars plant, quaternion winding, strict landing."""
import math,os
from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_conjugate
from wheel_humanoid_lab.assets import Q1_WHEEL_CUBEMARS_CFG, Q1_POSITION_JOINTS, Q1_WHEEL_JOINTS

@configclass
class Config(DirectRLEnvCfg):
    decimation=4
    episode_length_s=5.0
    action_space=24
    # gyro/gravity/vel/z(10), q/qd(44), actions(24), phase(3), winding/air(2)
    observation_space=83
    state_space=0
    sim=sim_utils.SimulationCfg(dt=.005,render_interval=4)
    scene=InteractiveSceneCfg(num_envs=64,env_spacing=4.,replicate_physics=True)
    robot=Q1_WHEEL_CUBEMARS_CFG.replace(prim_path='/World/envs/env_.*/Robot')
    contacts=ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*',history_length=3,update_period=.005)
    terrain=TerrainImporterCfg(prim_path='/World/ground',terrain_type='plane',collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=.9,restitution=0.))
    landing='wheels'
    # curriculum uses completed policy updates, never episode count
    curriculum_iteration=0

class Env(DirectRLEnv):
    def _setup_scene(self):
        self.robot=Articulation(self.cfg.robot)
        self.contacts=ContactSensor(self.cfg.contacts)
        self.cfg.terrain.num_envs=self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing=self.scene.cfg.env_spacing
        self.terrain=self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot']=self.robot
        self.scene.sensors['contacts']=self.contacts
        sim_utils.DomeLightCfg(intensity=1500.).func('/World/Light',sim_utils.DomeLightCfg(intensity=1500.))

    def __init__(self,cfg,**kwargs):
        # Local USD output: do not write into the original Q1 project.
        cfg.robot.spawn.usd_dir=str(Path(__file__).resolve().parent/'usd')
        cfg.robot.spawn.articulation_props.enabled_self_collisions=False
        # Symmetric 0.30-rad knee start; exact pelvis height measured by FK below.
        cfg.robot.init_state.joint_pos['.*_knee_joint']=.30
        cfg.robot.init_state.joint_pos['.*_hip_pitch_joint']=-.15
        cfg.robot.init_state.joint_pos['.*_knee_roller_joint']=.444444*.30
        cfg.robot.init_state.pos=(0.,0.,self._standing_height())
        super().__init__(cfg,**kwargs)
        self.pids,self.names=self.robot.find_joints(Q1_POSITION_JOINTS,preserve_order=True)
        self.wids,_=self.robot.find_joints(Q1_WHEEL_JOINTS,preserve_order=True)
        self.kids,_=self.robot.find_joints(['l_knee_joint','r_knee_joint'],preserve_order=True)
        self.rids,_=self.robot.find_joints(['l_knee_roller_joint','r_knee_roller_joint'],preserve_order=True)
        self.wb,_=self.contacts.find_bodies(['l_wheel_link','r_wheel_link'],preserve_order=True)
        self.rb,_=self.contacts.find_bodies(['l_knee_roller_link','r_knee_roller_link'],preserve_order=True)
        self.bad,_=self.contacts.find_bodies(['pelvis','torso','head_link','.*_thigh_link','.*_hip_pitch_link','.*_shoulder_.*','.*_elbow_.*','.*_wrist_.*','.*_gripper_.*'])
        n=self.num_envs; dev=self.device
        self.actions=torch.zeros(n,24,device=dev); self.previous=self.actions.clone()
        self.prev_quat=self.robot.data.root_quat_w.clone()
        self.winding=torch.zeros(n,device=dev); self.air=torch.zeros(n,device=dev)
        self.air_max=self.air.clone(); self.hold=self.air.clone(); self.peak=self.air.clone()
        self.jumped=torch.zeros(n,dtype=torch.bool,device=dev)
        self.failed=self.jumped.clone(); self.success=self.jumped.clone(); self.lifted=self.jumped.clone()
        self.tick=-1; self.updates=0
        self.home=self.robot.data.default_joint_pos[0,self.pids].clone()
        self.limits=self.robot.data.soft_joint_pos_limits[0,self.pids]
        self.reference,self.motion_meta=self._make_reference()
        self.metric_history=[]
        print('Q1_BACKFLIP_INIT',self.motion_meta,'height',cfg.robot.init_state.pos,'joints',self.names,flush=True)

    @staticmethod
    def _standing_height():
        # URDF FK for left wheel center, translation + rotation (no external kinematics dep).
        import xml.etree.ElementTree as ET
        root=ET.parse(Path(os.environ['WHEEL_HUMANOID_ROOT'])/'urdf/wheel_humanoid.urdf').getroot()
        joints={j.find('child').attrib['link']:j for j in root.findall('joint')}
        chain=[]; link='l_wheel_link'
        while link in joints:
            j=joints[link];chain.append(j);link=j.find('parent').attrib['link']
        T=np.eye(4)
        def rotation(axis,a):
            axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
            x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
        for j in reversed(chain):
            o=j.find('origin'); xyz=[0.,0.,0.] if o is None else list(map(float,o.get('xyz','0 0 0').split()))
            rpy=[0.,0.,0.] if o is None else list(map(float,o.get('rpy','0 0 0').split()))
            A=np.eye(4);A[:3,3]=xyz
            A[:3,:3]=rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0]);T=T@A
            q={'l_hip_pitch_joint':-.15,'l_knee_joint':.30}.get(j.attrib['name'],0.)
            if q:
                A=np.eye(4);A[:3,:3]=rotation(list(map(float,j.find('axis').attrib['xyz'].split())),q);T=T@A
        return float(.102-T[2,3])

    def _make_reference(self):
        src=Path(__file__).resolve().parent.parent/'reference_isaacsim_backflip/motions/backflip_g1.npz'
        d=np.load(src,allow_pickle=True); names=list(map(str,d['joint_names']));g=d['joint_pos']
        # Video-derived timing and sagittal channels, constrained to Q1 geometry.
        t=np.arange(251)/50.; ref=np.tile(self.home.cpu().numpy(),(251,1))
        st=np.arange(len(g))/float(d['fps'])
        takeoff=float(d['takeoff_time']);source_td=float(d['touchdown_time']);q1_td=1.30
        sample_t=np.where(t<=takeoff,t,np.where(t<q1_td,takeoff+(t-takeoff)*(source_td-takeoff)/(q1_td-takeoff),source_td+(t-q1_td)))
        mapping={}
        for side,long in [('l','left'),('r','right')]:
            for dst,suffix,sign in [('hip_pitch','hip_pitch',1),('knee','knee',1),('shoulder_pitch','shoulder_pitch',1),('elbow','elbow_pitch',-1)]:
                qn=f'{side}_{dst}_joint';gn=f'{long}_{suffix}_joint'
                if gn not in names:continue
                i=self.names.index(qn);channel=np.interp(sample_t,st,g[:,names.index(gn)])
                ref[:,i]+=sign*(channel-channel[0]);mapping[qn]={'source':gn,'sign':sign}
        # Last G1 pose uses hand support; replace with Q1 landing target after touchdown.
        td=q1_td;start=td
        target=self.home.cpu().numpy().copy()
        if self.cfg.landing=='kneel':
            for side in ['l','r']:
                target[self.names.index(f'{side}_knee_joint')]=2.20 if side=='l' else 1.85
                target[self.names.index(f'{side}_hip_pitch_joint')]=-1.02 if side=='l' else -.80
        blend=np.clip((t-start)/.3,0,1)
        ref=ref*(1-blend[:,None])+target*blend[:,None]
        lo=self.limits[:,0].cpu().numpy();hi=self.limits[:,1].cpu().numpy()
        clipped=np.clip(ref,lo+.02,hi-.02)
        # Retime sharp video edges to obey URDF velocity; targets remain hints, not forced root motion.
        vlim=self.robot.data.joint_vel_limits[0,self.pids].cpu().numpy()
        for k in range(1,len(clipped)):
            clipped[k]=clipped[k-1]+np.clip(clipped[k]-clipped[k-1],-vlim*.02*.8,vlim*.02*.8)
        meta={'source':str(src),'mapping':mapping,'takeoff':float(d['takeoff_time']),'touchdown':td,'source_touchdown':source_td,'landing':self.cfg.landing,'clipped_fraction':float(np.mean(ref!=np.clip(ref,lo+.02,hi-.02))),'fully_dynamic':True,'self_collisions':False}
        np.savez(Path(__file__).resolve().parent/f'reference_{self.cfg.landing}.npz',joint_pos=clipped,joint_names=self.names,fps=50.)
        return torch.tensor(clipped,device=self.device,dtype=torch.float32),meta

    def _phase(self):
        return self.episode_length_buf.float()*self.step_dt

    def _target(self):
        t=self._phase();idx=(t*50).long().clamp(0,len(self.reference)-1)
        ref=self.reference[idx]
        # First 300 updates stabilize, next 700 emphasize jump; gradually introduce reference flip.
        blend=max(0.,min(1.,(self.updates-300)/700))
        return self.home+blend*(ref-self.home)

    def _pre_physics_step(self,actions):
        if not torch.isfinite(actions).all():raise FloatingPointError('nonfinite actions')
        self.previous.copy_(self.actions);self.actions.copy_(actions.clamp(-3,3))
        self.target=(self._target()+.55*self.actions[:,:22]).clamp(self.limits[:,0],self.limits[:,1])

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target,joint_ids=self.pids)
        idx=(self._phase()*50).long().clamp(0,len(self.reference)-2)
        blend=max(0.,min(1.,(self.updates-300)/700))
        refvel=blend*(self.reference[idx+1]-self.reference[idx])/self.step_dt
        self.robot.set_joint_velocity_target(refvel,joint_ids=self.pids)
        self.robot.set_joint_velocity_target(25*self.actions[:,22:].clamp(-1,1),joint_ids=self.wids)
        self.robot.set_joint_position_target(.444444*self.robot.data.joint_pos[:,self.kids],joint_ids=self.rids)
        self.robot.set_joint_velocity_target(.444444*self.robot.data.joint_vel[:,self.kids],joint_ids=self.rids)

    def _state(self):
        if self.tick==self.common_step_counter:return
        self.tick=self.common_step_counter
        dq=quat_mul(self.robot.data.root_quat_w,quat_conjugate(self.prev_quat))
        dq=torch.where(dq[:,:1]<0,-dq,dq)
        norm=dq[:,1:].norm(dim=-1)
        rot=2*torch.atan2(norm,dq[:,0]);dy=rot*dq[:,2]/norm.clamp_min(1.e-8)
        self.delta=dy;self.winding+=dy;self.prev_quat.copy_(self.robot.data.root_quat_w)
        forces=self.contacts.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        self.wcontact=forces[:,self.wb]>5;self.rcontact=forces[:,self.rb]>5
        illegal=(forces[:,self.bad]>10).any(dim=1)
        self.failed|=illegal
        airborne=~((forces>5).any(dim=1))
        self.air=torch.where(airborne,self.air+self.step_dt,torch.zeros_like(self.air))
        self.air_max=torch.maximum(self.air_max,self.air)
        rising=(self.robot.data.root_lin_vel_w[:,2]>.3)&(self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]>self.cfg.robot.init_state.pos[2]+.04)
        self.lifted|=airborne&rising
        self.jumped|=(self.air>=.08)&self.lifted
        z=self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2];self.peak=torch.maximum(self.peak,z)
        upright=-self.robot.data.projected_gravity_b[:,2]
        support=self.wcontact.all(dim=1) if self.cfg.landing=='wheels' else (self.wcontact.any(dim=1)&self.rcontact.any(dim=1))
        stable=(upright>.94)&(self.robot.data.root_ang_vel_b.norm(dim=1)<1.0)&(self.robot.data.root_lin_vel_w[:,:2].norm(dim=1)<.3)
        full=(self.winding+2*math.pi).abs()<.25
        ok=full&self.jumped&support&stable&(~self.failed)&(self._phase()>self.motion_meta['touchdown'])
        self.hold=torch.where(ok,self.hold+self.step_dt,torch.zeros_like(self.hold));self.success|=self.hold>=2.

    def _get_observations(self):
        self._state();t=self._phase();r=self.robot.data
        phase=torch.stack((torch.sin(t*2*math.pi/5),torch.cos(t*2*math.pi/5),t/5),dim=-1)
        obs=torch.cat((r.root_ang_vel_b*.25,r.projected_gravity_b,r.root_lin_vel_b,r.root_pos_w[:,2:3]-self.scene.env_origins[:,2:3],r.joint_pos[:,self.pids]-self.home,r.joint_vel[:,self.pids]*.05,self.actions,phase,self.winding[:,None]/(2*math.pi),self.air_max[:,None]),dim=-1)
        if not torch.isfinite(obs).all():raise FloatingPointError('nonfinite observations')
        return {'policy':obs}

    def _get_rewards(self):
        self._state();r=self.robot.data;t=self._phase()
        level=max(0.,min(1.,(self.updates-300)/700))
        flight=(t>self.motion_meta['takeoff'])&(t<self.motion_meta['touchdown'])
        tracking=torch.exp(-((r.joint_pos[:,self.pids]-self._target())**2).mean(dim=-1)/.3)
        upright=(-r.projected_gravity_b[:,2]).clamp(0,1)
        stand=5*upright*self.wcontact.all(dim=1)*torch.exp(-r.root_ang_vel_b.square().sum(dim=-1))
        back=(-self.delta).clamp(0,.15)/self.step_dt
        progress=(-self.winding/(2*math.pi)).clamp(0,1)
        jump=(r.root_lin_vel_w[:,2]).clamp(0,3)*flight
        goal=(-2*math.pi*((t-self.motion_meta['takeoff'])/(self.motion_meta['touchdown']-self.motion_meta['takeoff'])).clamp(0,1))
        pitch_tracking=torch.exp(-((self.winding-goal)/.8).square())
        reward=(2*tracking+(1-level)*stand+level*(6*pitch_tracking+12*back*flight*(self.air>0)*self.lifted+8*progress*flight*(self.air>0)*self.lifted+2*jump+30*(self.hold>0).float()+100*self.success.float())-.02*(self.actions-self.previous).square().sum(dim=-1)-.005*self.actions.square().sum(dim=-1)-2.e-5*r.applied_torque.square().sum(dim=-1)-.2*r.root_ang_vel_b[:,[0,2]].square().sum(dim=-1)-50*self.failed.float())*self.step_dt
        if not torch.isfinite(reward).all():raise FloatingPointError('nonfinite reward')
        return reward

    def _get_dones(self):
        self._state()
        return self.failed|(self.robot.data.root_pos_w[:,2]-self.scene.env_origins[:,2]<.18)|((self.robot.data.root_pos_w[:,:2]-self.scene.env_origins[:,:2]).norm(dim=-1)>1.8),self.episode_length_buf>=self.max_episode_length-1

    def _reset_idx(self,ids):
        if ids is None:ids=self.robot._ALL_INDICES
        if hasattr(self,'success'):
            metrics={'success':self.success[ids].float().mean().item(),'airborne':self.jumped[ids].float().mean().item(),'backturn':(-self.winding[ids]).mean().item(),'peak_height':self.peak[ids].mean().item(),'illegal':self.failed[ids].float().mean().item(),'episode_steps':self.episode_length_buf[ids].float().mean().item(),'peak_current_a':max(a.current[ids].abs().max().item() for a in self.robot.actuators.values() if hasattr(a,'current')),'saturation':sum(a.saturated[ids].float().mean().item() for a in self.robot.actuators.values() if hasattr(a,'saturated'))/max(1,sum(hasattr(a,'saturated') for a in self.robot.actuators.values()))}
            self.extras['log']={f'Backflip/{k}':v for k,v in metrics.items()}
            self.metric_history.append(metrics)
            if len(self.metric_history)>1000:self.metric_history=self.metric_history[-1000:]
        self.robot.reset(ids);super()._reset_idx(ids)
        root=self.robot.data.default_root_state[ids].clone();root[:,:3]+=self.scene.env_origins[ids]
        q=self.robot.data.default_joint_pos[ids].clone();qd=torch.zeros_like(q)
        q[:,self.pids]+=(torch.rand(len(ids),22,device=self.device)-.5)*.02
        q[:,self.rids]=q[:,self.kids]*.444444
        self.robot.write_root_pose_to_sim(root[:,:7],ids);self.robot.write_root_velocity_to_sim(root[:,7:],ids)
        self.robot.write_joint_state_to_sim(q,qd,None,ids)
        if hasattr(self,'success'):
            for b in [self.winding,self.air,self.air_max,self.hold,self.peak]:b[ids]=0
            for b in [self.jumped,self.failed,self.success,self.lifted]:b[ids]=False
            self.actions[ids]=0;self.previous[ids]=0;self.prev_quat[ids]=root[:,3:7]
