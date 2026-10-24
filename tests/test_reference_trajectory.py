"""Actual extracted reference, full CAD FK and reverse-time invariants, CPU only."""
import importlib.util,json,sys,tempfile
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('video_trajectory',ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/native_reference/trajectory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from tools.audit_native_pose_geometry import fk,rotation
import xml.etree.ElementTree as ET
import numpy as np
REF=ROOT/'data/reference/native_unbox_20261024/reference.json';URDF=ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf'

class TrajectoryTest(unittest.TestCase):
 def setUp(self):self.data=json.loads(REF.read_text());self.t=m.VideoTrajectory(REF,self.data['joint_names'],URDF)
 def test_real_mediapipe_source_and_all_eight_seconds_retained(self):
  self.assertTrue(self.data['source']['mediapipe_executed']);self.assertEqual(self.t.duration,8.);self.assertGreater(len(self.t.times),100)
  self.assertTrue(self.data['quality']['non_monotonic_source_retained']);self.assertLess(float(self.t.height.min()),float(self.t.height[0])-.03)
 def test_reverse_is_same_motion_in_opposite_time(self):
  p=torch.linspace(0,1,51)
  for a,b in zip(self.t.sample(p,reverse=True),self.t.sample(1-p)):torch.testing.assert_close(a,b)
 def test_full_fk_both_wheels_2mm_above_ground_including_root_lean(self):
  urdf=ET.parse(URDF).getroot()
  for i in (0,30,65,100,len(self.t.q)-1):
   links=fk(urdf,dict(zip(self.t.joint_names,self.t.q[i].tolist())));pitch=float(self.t.pitch[i]);R=rotation([0,1,0],pitch)
   for side in ('left','right'):
    wheel=(R@links[side+'_wheel_link'])[:3,3]
    self.assertAlmostEqual(float(self.t.height[i])+wheel[2]-.1,.002,places=5)
 def test_terminal_has_microbend_and_waist_does_not_bend_past_cad_stop(self):
  index={n:i for i,n in enumerate(self.t.joint_names)}
  self.assertLess(abs(float(self.t.q[-1,index['left_knee_1_Pitch_joint']])),.42)
  self.assertTrue(bool((self.t.q[:,index['center_waist_Pitch_joint']]>=0).all()))
 def test_wrong_video_and_unapproved_data_fail_closed(self):
  for key in ('video','quality'):
   data=json.loads(REF.read_text())
   if key=='video':data['source']['video_id']='unrelated'
   else:data['quality']['approved_for_training']=False
   with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'ref.json';p.write_text(json.dumps(data))
    with self.assertRaises(ValueError):m.VideoTrajectory(p,data['joint_names'],URDF)
 def test_legacy_joint_order_is_rejected(self):
  with self.assertRaises(ValueError):m.VideoTrajectory(REF,list(reversed(self.t.joint_names)),URDF)
if __name__=='__main__':unittest.main()

class BootResetTest(unittest.TestCase):
 def test_reference_web_reset_physically_starts_at_first_crouch_not_reverse_stand(self):
  import ast
  from types import SimpleNamespace
  data=json.loads(REF.read_text());trajectory=m.VideoTrajectory(REF,data['joint_names'],URDF)
  urdf=ET.parse(URDF).getroot();mimics=[j for j in urdf.findall('joint') if j.find('mimic') is not None]
  names=data['joint_names']+[j.get('name') for j in mimics];saved={}
  root=torch.zeros(1,13);root[:,3]=1.
  robot=SimpleNamespace(_ALL_INDICES=torch.tensor([0]),joint_names=names,data=SimpleNamespace(default_joint_pos=torch.zeros(1,len(names)),default_root_state=root),
   write_root_pose_to_sim=lambda values,env_ids:saved.update(root=values.clone()),write_root_velocity_to_sim=lambda *a,**k:None,write_joint_state_to_sim=lambda values,*a,**k:saved.update(q=values.clone()))
  class Scene(dict):env_origins=torch.zeros(1,3)
  skill=SimpleNamespace(trajectory=trajectory,pending_reset_phase=torch.zeros(1),level=1.)
  env=SimpleNamespace(scene=Scene(robot=robot),device='cpu',command_manager=SimpleNamespace(get_term=lambda _:skill),cfg=SimpleNamespace(reference_skill='kneel',boot_squat=True,reference_phase_resets=False))
  path=ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/native_reference/mdp.py'
  node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='reset_reference')
  ns=dict(torch=torch,Q1_INERTIA_JOINT_ORDER=data['joint_names'],Q1_INERTIA_MIMIC_JOINTS=[j.get('name') for j in mimics],Q1_INERTIA_MIMIC_DRIVERS=[j.find('mimic').get('joint') for j in mimics],Q1_INERTIA_MIMIC_MULTIPLIERS=[float(j.find('mimic').get('multiplier')) for j in mimics])
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);ns['reset_reference'](env,torch.tensor([0]))
  torch.testing.assert_close(saved['q'][0,:23],trajectory.q[0]);self.assertAlmostEqual(float(saved['root'][0,2]),float(trajectory.height[0]),places=6)
