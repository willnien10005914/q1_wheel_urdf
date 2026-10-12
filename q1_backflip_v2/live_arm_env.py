"""Arms-down backflip fine-tuning from physically settled web standing states.
Only episode Reset differs from arm_env; policy and actuator dynamics are identical.
"""
import json,os
from pathlib import Path
import torch
from arm_env import Env as ArmEnv,Config,DEFAULT,LOW,HIGH
class Env(ArmEnv):
 def __init__(self,cfg,**kwargs):
  super().__init__(cfg,**kwargs)
  path=Path(__file__).resolve().parents[1]/'docs/training/backflip_standing_bank.json'
  bank=json.loads(path.read_text());self.standing_bank_path=path
  order=[bank['joint_names'].index(n) for n in self.robot.joint_names]
  self.bank_q=torch.tensor([x['joint_position'] for x in bank['states']],device=self.device)[:,order]
  self.bank_qd=torch.tensor([x['joint_velocity'] for x in bank['states']],device=self.device)[:,order]
  self.bank_root=torch.tensor([x['root_pose']+x['root_velocity'] for x in bank['states']],device=self.device)
 def _reset_idx(self,ids):
  super()._reset_idx(ids)
  if not hasattr(self,'bank_q'):return
  if ids is None:ids=self.robot._ALL_INDICES
  selected=ids[torch.rand(len(ids),device=self.device)<.7]
  if not len(selected):return
  i=torch.randint(len(self.bank_q),(len(selected),),device=self.device)
  root=self.bank_root[i].clone();root[:,:3]+=self.scene.env_origins[selected]
  self.robot.write_joint_state_to_sim(self.bank_q[i],self.bank_qd[i],env_ids=selected)
  self.robot.write_root_pose_to_sim(root[:,:7],env_ids=selected)
  self.robot.write_root_velocity_to_sim(root[:,7:],env_ids=selected)
  self.prev_quat[selected]=root[:,3:7]
