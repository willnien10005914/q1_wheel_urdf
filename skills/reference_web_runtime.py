"""Web motion selection for video-reference policies; no simulator boot at import."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from skills.policy import FrozenActor

CONTRACT='q1-native-video-v1-23x90'
SKILLS=('standup','kneel','walk','backflip','skate')
ROOT=Path(__file__).resolve().parents[1]


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default


class ReferenceWeb:
    def __init__(self,env,models,training_state,allow_preview=False):
        self.env=env;self.robot=env.scene['robot'];self.skill=env.command_manager.get_term('skill')
        self.motors=env.action_manager.get_term('motors');self.models_root=Path(models)
        self.training_state=Path(training_state);self.allow_preview=allow_preview
        self.command=torch.zeros(1,3,device=env.device);self.mode='kneel_hold'
        self.actors={};self.models={};self.errors={};self.training={};self.last_refresh=0.
        self.failed=False;self.message='影片起始蹲姿；等待已驗證的起立／蹲下模型。'
        self.feedback_until=0.
        self.last_completed=None
        self.skill.hold_pose('kneel')
        trajectory=self.skill.trajectory
        self.reference_sha256=trajectory.sha256;self.urdf_sha256=trajectory.urdf_sha256
        self.refresh()

    def refresh(self):
        if time.monotonic()-self.last_refresh<2.:return
        self.last_refresh=time.monotonic();self.training=read_json(self.training_state)
        for name in SKILLS:
            folder=self.models_root/name;path=folder/'model.pt';metadata=folder/'validated.json'
            if not path.exists():path=folder/'latest.pt';metadata=folder/'latest.json'
            if not path.exists():
                self.actors.pop(name,None);self.models.pop(name,None);continue
            try:
                metadata_bytes=metadata.read_bytes()
                fingerprint=(path.name,path.stat().st_mtime_ns,path.stat().st_size,hashlib.sha256(metadata_bytes).hexdigest())
                if self.models.get(name,{}).get('fingerprint')==fingerprint:continue
                info=json.loads(metadata_bytes);digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if (info.get('contract')!=CONTRACT or info.get('skill')!=name
                    or info.get('sha256')!=digest or info.get('reference_sha256')!=self.reference_sha256
                    or info.get('urdf_sha256')!=self.urdf_sha256):
                    raise ValueError('模型／影片軌跡／URDF 契約不符')
                validated=info.get('validated') is True and path.name=='model.pt'
                if path.name=='model.pt' and not validated:raise ValueError('正式模型缺少通過驗證標記')
                actor=FrozenActor(path,self.env.device)
                if actor.input_dim!=90 or actor.mlp[-1].out_features!=23:raise ValueError('需要 23 action／90 observation')
                self.actors[name]=actor
                self.models[name]=dict(path=str(path),sha256=digest,validated=validated,iteration=actor.training_iteration,
                    contract=CONTRACT,skill=name,reference_sha256=self.reference_sha256,urdf_sha256=self.urdf_sha256,fingerprint=fingerprint)
                self.errors.pop(name,None)
            except (OSError,ValueError,RuntimeError,KeyError) as error:
                self.actors.pop(name,None);self.models.pop(name,None);self.errors[name]=str(error)

    def ready(self,name):
        return name in self.actors and (self.models[name]['validated'] or self.allow_preview)

    def policy_key(self):
        return {'stand_hold':'standup','kneel_hold':'kneel','slide':'skate'}.get(self.mode,self.mode)

    def feedback(self,message):
        self.message=message;self.feedback_until=time.monotonic()+6.

    def settle_completed(self):
        """Keep the endpoint policy running after a one-shot physically succeeds."""
        if not bool(getattr(self.skill,'success',torch.tensor([False]))[0]):return
        if self.mode=='standup':self.last_completed='standup';self.mode='stand_hold'
        elif self.mode=='kneel':self.last_completed='kneel';self.mode='kneel_hold'
        elif self.mode=='backflip':
            self.last_completed='backflip';self.skill.hold_pose('stand');self.mode='stand_hold'

    def resolve_target(self,name):
        target={'standard':'stand','stand':'stand','unbox':'standup','getup':'standup','standup':'standup',
            'kneel':'kneel','walk':'walk','skate':'walk','slide':'slide','sliding':'slide','backflip':'backflip'}.get(name)
        if target=='stand' and self.mode=='kneel_hold':target='standup'
        if target=='standup' and self.mode in ('stand_hold','walk','slide'):target='stand'
        return target

    def action_reason(self,name):
        target=self.resolve_target(name)
        if target is None:return '舊正反躺 Recovery 模型已保留，但尚未通過本版 URDF／策略契約驗證，此頁暫停使用。'
        key='standup' if target=='stand' else 'skate' if target=='slide' else target
        if not self.ready(key):return f'{key} 模型尚未通過影片／物理驗證；等待訓練結果。'
        if self.failed:return '動作檢查失敗，請先 Reset sim。'
        if self.mode in ('standup','kneel','backflip'):return '目前動作尚未完成並站穩／蹲穩，請等待完成。'
        if target=='standup' and self.mode!='kneel_hold':return '請先完成慢蹲，再從蹲姿起立。'
        if target in ('kneel','walk','slide','backflip') and self.mode=='kneel_hold':return '目前是蹲姿，請先按 Stand up 起身並等待站穩。'
        if target in ('kneel','backflip'):
            velocity=getattr(self.robot.data,'root_lin_vel_b',None)
            angular=getattr(self.robot.data,'root_ang_vel_b',None)
            moving=(velocity is not None and float(velocity[0].norm())>=.25)
            moving|=angular is not None and float(angular[0].norm())>=.6
            if moving or bool(self.command.abs().max()>.01):return '請先停止行進並站穩，再執行慢蹲或後空翻。'
        return ''

    def request(self,name):
        self.settle_completed()
        if name=='reset':
            self.env.reset();self.skill.hold_pose('kneel');self.mode='kneel_hold';self.command.zero_();self.failed=False
            self.last_completed=None
            self.feedback('已重設為影片起始蹲姿。');return True
        target=self.resolve_target(name);reason=self.action_reason(name)
        if reason:self.feedback(reason);return False
        key='standup' if target=='stand' else 'skate' if target=='slide' else target
        before_root=self.robot.data.root_state_w.clone();before_q=self.robot.data.joint_pos.clone()
        if target=='stand':self.skill.hold_pose('stand');self.mode='stand_hold'
        else:self.skill.set_mode(target);self.mode=target
        if not torch.equal(before_root,self.robot.data.root_state_w) or not torch.equal(before_q,self.robot.data.joint_pos):
            raise RuntimeError('Skill selection wrote physical pose')
        self.command.zero_();self.failed=False;self.motors.manual={}
        self.last_completed=None
        if target=='slide':self.command[:,0]=.3
        self.feedback(f'{target} · '+('已驗證影片 PPO' if self.models[key]['validated'] else '實驗 preview，尚未通過驗證'))
        return True

    def drive(self,values):
        self.settle_completed()
        try:v=[float(values.get(k,0.)) for k in ('vx','vy','yaw')]
        except (ValueError,TypeError):return False
        if not all(math.isfinite(x) for x in v):return False
        if self.mode in ('kneel','kneel_hold','standup','backflip'):return False
        if self.mode not in ('walk','slide') and any(abs(x)>1e-6 for x in v):
            if not self.request('walk'):return False
        if self.mode=='slide':v=[max(0.,min(.4,v[0])),0.,max(-.1,min(.1,v[2]))]
        else:v=[max(-.2,min(.4,v[0])),0.,max(-.15,min(.15,v[2]))]
        self.command[0]=torch.tensor(v,device=self.env.device);return True

    def action(self):
        self.refresh();self.settle_completed();term=self.env.command_manager.get_term('base_velocity')
        term.command[:]=self.command;term.time_left.fill_(1.e9);term.is_standing_env.fill_(not bool(self.command.abs().max()>0))
        self.skill._update_command();obs=self.env.observation_manager.compute()['policy']
        key=self.policy_key()
        if not self.ready(key):return torch.zeros(1,23,device=self.env.device)
        return self.actors[key](obs)

    def snapshot(self):
        s=self.skill;s.update_state();r=self.robot.data
        self.settle_completed()
        pose={'walk':'skate','standup':'unbox','stand_hold':'unbox_hold','kneel_hold':'kneel'}.get(self.mode,self.mode)
        success=bool(s.success[0]);phase=float(s.normalized_phase[0])
        if self.mode=='kneel':pose='kneel' if success else 'kneeling'
        if self.mode=='standup' and success:pose='unbox_hold'
        self.failed|=bool(s.illegal[0])
        finite=bool(torch.isfinite(r.root_state_w).all() and torch.isfinite(r.joint_pos).all())
        gravity=getattr(r,'projected_gravity_b',None)
        if not finite or (gravity is not None and self.mode!='backflip' and float(-gravity[0,2])<.45):self.failed=True
        quality=getattr(s,'slow_kneel',None)
        if quality is not None:self.failed|=bool(quality.failed[0])
        key=self.policy_key();model=self.models.get(key,{})
        label='已驗證影片 PPO' if model.get('validated') else '實驗 preview，未通過驗證' if model else '等待模型／reference 控制'
        if time.monotonic()>=self.feedback_until:self.message=f'{self.mode} · {label}'
        message=self.message+(' · 動作／接觸檢查失敗，請 Reset sim' if self.failed else '')
        q=r.joint_pos[0].tolist();qd=r.joint_vel[0].tolist();tau=r.applied_torque[0].tolist()
        local=r.root_pos_w[0]-self.env.scene.env_origins[0];quat=r.root_quat_w[0].tolist()
        policies={k:self.ready(k) for k in SKILLS};policies['unbox']=policies['standup'];policies['slide']=policies['skate']
        cycles=getattr(s,'cycles',None)
        names=('stand','skate','walk','kneel','unbox','backflip','slide')
        reasons={name:self.action_reason(name) for name in names}
        slow_kneel={} if quality is None else {name:float(getattr(quality,name)[0]) for name in
            ('contact_fraction','max_bilateral_loss_s','max_air_s','max_descent_mps','max_fast_descent_s','max_hop_s')}
        if quality is not None:slow_kneel['failed']=bool(quality.failed[0])
        return dict(ok=True,contract=CONTRACT,environment='ManagerBasedRLEnv',urdf='urdf/q1_inertia/quanta_robot_wheeled.urdf',
            action_dim=23,observation_dim=90,mass_kg=float(r.default_mass[0].sum()),pose=pose,message=message,
            cmd=dict(vx=float(self.command[0,0]),vy=0.,yaw=float(self.command[0,2])),
            base=dict(x=float(local[0]),y=float(local[1]),z=float(local[2]),qw=quat[0],qx=quat[1],qy=quat[2],qz=quat[3]),
            joints={n:dict(pos=q[i],vel=qd[i],torque=tau[i]) for i,n in enumerate(self.robot.joint_names)},
            policies=policies,models=self.models,model_errors=self.errors,
            supported_actions={name:not reason for name,reason in reasons.items()},action_unavailable_reasons=reasons,
            recovery_available=False,recovery_message='舊正反躺 Recovery 模型已保留；本輪不重訓，尚未通過本版 URDF／策略契約驗證，按鈕暫停使用。',
            manual_control_available=False,boot_pose='video_kneel',reference_sha256=self.reference_sha256,
            motion_phase=phase,last_completed_motion=self.last_completed,backflip=dict(success=(success and self.mode=='backflip') or self.last_completed=='backflip',hold_s=float(s.best_hold[0]),
                winding_rad=float(s.winding[0]),airtime_s=float(s.max_air_time[0]),illegal=bool(s.illegal[0])),
            slow_kneel=slow_kneel,kneel_duration_s=getattr(s,'kneel_duration',None),
            sliding=dict(active=self.mode=='slide',cycles=cycles.cycles[0].tolist() if cycles else [0,0]),training=self.training)
