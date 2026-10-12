"""Search bounded motor targets on real landings; never rewrite state during a trial."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=256);p.add_argument('--rounds',type=int,default=5);p.add_argument('--steps',type=int,default=450);p.add_argument('--out',required=True);p.add_argument('--validate-candidates')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from skills.interactive_env import Env,Config
class PerEnv:
 def __init__(self,value):self.value=value
 def __getitem__(self,index):return self.value
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=4107;cfg.sim.device=args.device or 'cuda:0'
env=Env(cfg,ROOT);env.reset();runtime=env.recoveries['boot_kneel'];s=runtime.state;env.recovery=runtime
runtime.actor=lambda obs:torch.zeros(env.num_envs,24,device=env.device)
bank=json.loads((ROOT/'docs/training/backflip_landing_bank.json').read_text());order=[bank['joint_names'].index(n) for n in env.robot.joint_names]
q=torch.tensor([x['joint_position'] for x in bank['states']],device=env.device)[:,order];qd=torch.tensor([x['joint_velocity'] for x in bank['states']],device=env.device)[:,order];root=torch.tensor([x['root_pose']+x['root_velocity'] for x in bank['states']],device=env.device)
base_values=s.stand_values[0].clone();base_target=s.targets[0,4].clone()
# duration, wheel kp/kd/bias/vx gain, final hip/knee/waist
lo=torch.tensor([1.,30.,3.,-.25,0.,-.55,.3,-.15],device=env.device)
hi=torch.tensor([5.,180.,35.,.1,45.,-.1,1.0,.35],device=env.device)
mean=torch.tensor([3.,101.,19.5,-.086,26.,-.35,.7,.1],device=env.device);std=(hi-lo)*.25
history=[];best_score=-1e9;best=None;ids=torch.arange(env.num_envs,device=env.device)
with torch.inference_mode():
 for round_id in range(args.rounds):
  parameters=(mean+torch.randn(env.num_envs,len(mean),device=env.device)*std).clamp(lo,hi);parameters[0]=mean
  if args.validate_candidates:
   choices=json.loads(Path(args.validate_candidates).read_text())['history']
   parameters=torch.tensor([choices[i%len(choices)]['parameters'] for i in range(env.num_envs)],device=env.device)
  # Shared bank sample per candidate round, repeated seeds for physical diversity.
  index=torch.randint(len(q),(env.num_envs,),device=env.device) if args.validate_candidates else torch.arange(env.num_envs,device=env.device)%len(q);initial=root[index].clone();initial[:,:3]+=env.scene.env_origins
  env.robot.write_joint_state_to_sim(q[index],qd[index],env_ids=ids);env.robot.write_root_pose_to_sim(initial[:,:7],env_ids=ids);env.robot.write_root_velocity_to_sim(initial[:,7:],env_ids=ids)
  runtime.start_kneel();env.mode='unbox';s.stage.fill_(4);s.elapsed.zero_();s.hold.zero_();s.upright_active.fill_(False)
  values=base_values.expand(env.num_envs,-1).clone();values[:,11]=0.
  for col,param in [(5,0),(6,1),(7,2),(8,3),(12,4)]:values[:,col]=parameters[:,param]
  s.stand_values=PerEnv(values);s.stand_sit_back=PerEnv(torch.zeros(env.num_envs,dtype=torch.bool,device=env.device))
  target=base_target.expand(env.num_envs,-1).clone()
  for side in ['l','r']:
   target[:,env.robot.joint_names.index(side+'_hip_pitch_joint')]=parameters[:,5];target[:,env.robot.joint_names.index(side+'_knee_joint')]=parameters[:,6]
  target[:,env.robot.joint_names.index('waist_pitch_joint')]=parameters[:,7]
  s.targets=PerEnv(target)
  hold=torch.zeros(env.num_envs,device=env.device);height=hold.clone();valid=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
  for step in range(args.steps):
   env.step(env.actor_action());s.update();z=s.height
   valid&=(z>.15)&((env.robot.data.root_pos_w-env.scene.env_origins)[:,:2].norm(dim=-1)<3.)
   hold=torch.maximum(hold,torch.where(valid,s.stand_hold,0.));height=torch.maximum(height,torch.where(valid,z,0.))
  score=100*(hold>=1.).float()+10*hold+height+valid.float()
  top=score.topk(max(8,env.num_envs//8)).indices
  mean=parameters[top].mean(0);std=parameters[top].std(0).clamp(min=(hi-lo)*.025)
  i=int(score.argmax());record={'round':round_id,'successes':int((hold>=1.).sum()),'best_score':float(score[i]),'hold':float(hold[i]),'parameters':parameters[i].tolist()};
  if args.validate_candidates:
   record['groups']=[{'parameters':c['parameters'],'trials':int((ids%len(choices)==j).sum()),'successes':int(((hold>=1.)&(ids%len(choices)==j)).sum()),'stable_at_end':int(((s.stand_hold>=1.)&valid&(ids%len(choices)==j)).sum())} for j,c in enumerate(choices)]
  history.append(record);print('RISE_SEARCH',json.dumps(record),flush=True)
  if float(score[i])>best_score:best_score=float(score[i]);best=parameters[i].tolist()
  Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps({'history':history,'best':best,'physical_motor_only':True},indent=2))
env.close();app.close()
