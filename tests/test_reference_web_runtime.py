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
    _update_command=lambda :None)
  self.velocity=SimpleNamespace(command=torch.zeros(1,3),time_left=torch.zeros(1),is_standing_env=torch.ones(1,dtype=torch.bool))
  self.robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(1,13),joint_pos=torch.zeros(1,27)))
  self.env=SimpleNamespace(device='cpu',scene={'robot':self.robot},command_manager=SimpleNamespace(get_term=lambda name:self.skill if name=='skill' else self.velocity),
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

if __name__=='__main__':unittest.main()
