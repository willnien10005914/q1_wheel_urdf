"""CPU tests of real Web selection/gates; policy-file inference is stubbed."""
import hashlib,json,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from skills.reference_web_runtime import ReferenceWeb,CONTRACT

class ReferenceWebTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.calls=[]
  self.skill=SimpleNamespace(trajectory=SimpleNamespace(sha256='reference',urdf_sha256='urdf'),
    hold_pose=lambda name:self.calls.append(('hold',name)),set_mode=lambda name:self.calls.append(('mode',name)),
    _update_command=lambda :None,update_state=lambda :None,success=torch.tensor([False]),illegal=torch.tensor([False]),
    normalized_phase=torch.tensor([0.]),best_hold=torch.tensor([0.]),winding=torch.tensor([0.]),max_air_time=torch.tensor([0.]))
  self.velocity=SimpleNamespace(command=torch.zeros(1,3),time_left=torch.zeros(1),is_standing_env=torch.ones(1,dtype=torch.bool))
  self.robot=SimpleNamespace(joint_names=[str(i) for i in range(27)],data=SimpleNamespace(root_state_w=torch.zeros(1,13),joint_pos=torch.zeros(1,27),
    root_lin_vel_b=torch.zeros(1,3),root_ang_vel_b=torch.zeros(1,3),joint_vel=torch.zeros(1,27),applied_torque=torch.zeros(1,27),
    root_pos_w=torch.zeros(1,3),root_quat_w=torch.tensor([[1.,0.,0.,0.]]),default_mass=torch.ones(1,27)))
  class Scene(dict):env_origins=torch.zeros(1,3)
  self.env=SimpleNamespace(device='cpu',scene=Scene(robot=self.robot),command_manager=SimpleNamespace(get_term=lambda name:self.skill if name=='skill' else self.velocity),
    action_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(manual={})),observation_manager=SimpleNamespace(compute=lambda :{'policy':torch.zeros(1,90)}),reset=lambda :self.calls.append(('reset',)))
  class Actor:
   def __init__(self,path,device):self.input_dim=90;self.mlp=[SimpleNamespace(out_features=23)];self.training_iteration=2
   def __call__(self,obs):return torch.zeros(1,23)
  self.patch=patch('skills.reference_web_runtime.FrozenActor',Actor);self.patch.start()
  self.web=ReferenceWeb(self.env,self.root/'models',self.root/'status.json')
 def tearDown(self):self.patch.stop();self.tmp.cleanup()
 def model(self,name='standup',validated=True,**kwargs):
  folder=self.root/'models'/name;folder.mkdir(parents=True,exist_ok=True)
  path=folder/('model.pt' if validated else 'latest.pt');path.write_bytes(b'TEST ONLY actor')
  info=dict(contract=CONTRACT,skill=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_sha256='reference',urdf_sha256='urdf',validated=validated)
  info.update(kwargs);meta=folder/('validated.json' if validated else 'latest.json');meta.write_text(json.dumps(info))
  self.web.last_refresh=0.;self.web.refresh();return meta
 def test_boot_holds_video_kneel_without_reset_or_pose_writes(self):
  self.assertEqual(self.web.mode,'kneel_hold');self.assertEqual(self.calls,[('hold','kneel')])
  torch.testing.assert_close(self.robot.data.joint_pos,torch.zeros(1,27))
  self.assertFalse(self.web.drive({'vx':1.2}))
 def test_old_shape_compatible_policy_is_rejected(self):
  self.model(contract='q1-native-manager-v2-23x90')
  self.assertFalse(self.web.request('unbox'));self.assertEqual(self.web.mode,'kneel_hold')
 def test_reference_and_urdf_are_checked_not_only_action_shape(self):
  for key in ('reference_sha256','urdf_sha256'):
   self.model(**{key:'different'});self.assertFalse(self.web.ready('standup'))
 def test_fresh_rise_command_only_changes_manager_mode(self):
  self.model();q=self.robot.data.joint_pos.clone();root=self.robot.data.root_state_w.clone()
  self.assertTrue(self.web.request('stand'));self.assertEqual(self.calls[-1],('mode','standup'))
  torch.testing.assert_close(q,self.robot.data.joint_pos);torch.testing.assert_close(root,self.robot.data.root_state_w)
 def test_kneel_hold_does_not_borrow_standup_policy(self):
  self.model();self.web.actors['standup']=lambda obs:(_ for _ in ()).throw(AssertionError('wrong actor'))
  torch.testing.assert_close(self.web.action(),torch.zeros(1,23))
 def test_preview_requires_explicit_experimental_option(self):
  self.model(validated=False);self.assertFalse(self.web.ready('standup'))
  self.web.allow_preview=True;self.assertTrue(self.web.ready('standup'))
 def test_metadata_change_invalidates_previously_loaded_actor(self):
  meta=self.model();self.assertTrue(self.web.ready('standup'))
  info=json.loads(meta.read_text());info['reference_sha256']='tampered';meta.write_text(json.dumps(info));self.web.last_refresh=0.;self.web.refresh()
  self.assertFalse(self.web.ready('standup'))
 def test_validated_walk_command_is_bounded(self):
  self.model('walk');self.web.mode='stand_hold'
  self.assertTrue(self.web.drive({'vx':1.2,'yaw':.5}));self.assertEqual(self.web.mode,'walk')
  self.assertAlmostEqual(float(self.web.command[0,0]),.4);self.assertAlmostEqual(float(self.web.command[0,2]),.15)
 def test_recovery_is_not_silently_rerouted(self):
  self.model();self.assertFalse(self.web.request('recovery_supine_elbow'));self.assertEqual(self.web.mode,'kneel_hold')
  self.assertIn('--legacy-native',self.web.snapshot()['message'])
 def test_crouched_robot_cannot_flip_or_drive_even_with_validated_models(self):
  for name in ('backflip','kneel','walk','skate'):self.model(name)
  for name in ('backflip','kneel','walk','slide'):
   self.assertFalse(self.web.request(name));self.assertEqual(self.web.mode,'kneel_hold')
  status=self.web.snapshot();self.assertFalse(status['supported_actions']['backflip']);self.assertIn('先按 Stand up',status['message'])
 def test_rising_motion_cannot_be_interrupted(self):
  for name in ('standup','kneel','walk','backflip'):self.model(name)
  self.assertTrue(self.web.request('unbox'))
  for name in ('stand','kneel','walk','backflip'):
   self.assertFalse(self.web.request(name));self.assertEqual(self.web.mode,'standup')
 def test_completed_rise_enters_hold_and_allows_drive(self):
  self.model();self.model('walk');self.assertTrue(self.web.request('unbox'))
  self.skill.success[0]=True
  self.assertTrue(self.web.drive({'vx':.3}));self.assertEqual(self.web.mode,'walk')
 def test_standup_while_standing_holds_instead_of_restarting_crouch_reference(self):
  self.model();self.web.mode='stand_hold';self.assertTrue(self.web.request('unbox'))
  self.assertEqual(self.calls[-1],('hold','stand'));self.assertEqual(self.web.mode,'stand_hold')
 def test_slow_kneel_requires_stopped_motion(self):
  self.model('kneel');self.web.mode='walk';self.web.command[0,0]=.3
  self.assertFalse(self.web.request('kneel'));self.assertIn('先停止',self.web.message)
  self.web.command.zero_();self.assertTrue(self.web.request('kneel'))
 def test_rejection_feedback_survives_snapshots_then_expires(self):
  self.web.request('unbox');message=self.web.message
  self.assertEqual(self.web.snapshot()['message'],message)
  self.web.feedback_until=0.;self.assertNotEqual(self.web.snapshot()['message'],message)
 def test_kneel_quality_failure_disables_motion_until_reset(self):
  self.model();self.skill.slow_kneel=SimpleNamespace(failed=torch.tensor([True]),contact_fraction=torch.tensor([.4]),
    max_bilateral_loss_s=torch.tensor([.5]),max_air_s=torch.tensor([.1]),max_descent_mps=torch.tensor([1.]),
    max_fast_descent_s=torch.tensor([.3]),max_hop_s=torch.tensor([0.]))
  status=self.web.snapshot();self.assertTrue(status['slow_kneel']['failed'])
  self.assertFalse(status['supported_actions']['unbox']);self.assertFalse(self.web.request('unbox'))
 def test_successful_flip_returns_to_stand_hold_without_pose_write(self):
  self.web.mode='backflip';self.skill.success[0]=True
  q=self.robot.data.joint_pos.clone();root=self.robot.data.root_state_w.clone()
  status=self.web.snapshot()
  self.assertEqual(self.web.mode,'stand_hold');self.assertTrue(status['backflip']['success'])
  self.assertEqual(status['last_completed_motion'],'backflip');self.assertEqual(self.calls[-1],('hold','stand'))
  torch.testing.assert_close(q,self.robot.data.joint_pos);torch.testing.assert_close(root,self.robot.data.root_state_w)

if __name__=='__main__':unittest.main()
