"""Exercise the actual Web class without starting Kit; actor I/O is stubbed."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'scripts/reinforcement_learning/rsl_rl/play_native_web.py'
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('NativeWeb','read_json')]


class WebSlidingGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        folder=self.root/'models/slide';folder.mkdir(parents=True);self.folder=folder
        def actor(path,device):
            return SimpleNamespace(input_dim=90,mlp=[SimpleNamespace(out_features=23)],training_iteration=2,
                                   checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.ns=dict(Path=Path,json=json,time=time,torch=torch,math=__import__('math'),ROOT=self.root,
                     CONTRACT='q1-native-manager-v2-23x90',SLIDE_CONTRACT='q1-native-slide-v1-23x90',
                     FrozenActor=actor,a=SimpleNamespace(models=self.root/'models',training_state=self.root/'none.json'))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),self.ns)
        cls=self.ns['NativeWeb'];self.web=cls.__new__(cls)
        self.web.env=SimpleNamespace(device='cpu');self.web.last_refresh=0.
        self.web.models={};self.web.actors={};self.web.training={}
        self.web.mode='stand';self.web.command=torch.zeros(1,3)
        self.web.robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(1,13),joint_pos=torch.zeros(1,27)))
        self.calls=[];self.web.skill=SimpleNamespace(set_mode=self.calls.append,level=1.)
        self.web.motors=SimpleNamespace(manual={})

    def tearDown(self):self.tmp.cleanup()

    def model(self,validated=True,contract='q1-native-slide-v1-23x90'):
        path=self.folder/('model.pt' if validated else 'latest.pt');path.write_bytes(b'stub actor file')
        meta=dict(contract=contract,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validated=validated)
        (self.folder/('validated.json' if validated else 'latest.json')).write_text(json.dumps(meta))
        self.web.refresh()

    def test_shape_compatible_stand_model_cannot_masquerade_as_sliding(self):
        self.model(contract='q1-native-manager-v2-23x90')
        self.assertNotIn('slide',self.web.actors)
        self.assertFalse(self.web.request('slide'));self.assertEqual(self.calls,[])

    def test_training_preview_is_not_executable(self):
        self.model(validated=False)
        self.assertIn('slide',self.web.actors)
        self.assertFalse(self.web.request('slide'));self.assertEqual(self.web.mode,'stand')

    def test_validated_gait_changes_command_without_writing_robot_pose(self):
        self.model();before=self.web.robot.data.root_state_w.clone()
        self.assertTrue(self.web.request('slide'));self.assertEqual(self.calls,['slide'])
        self.assertEqual(self.web.mode,'slide');self.assertAlmostEqual(float(self.web.command[0,0]),.4)
        torch.testing.assert_close(self.web.robot.data.root_state_w,before)

    def test_keyboard_drive_stays_in_sliding_training_range(self):
        self.model();self.web.request('slide')
        self.assertTrue(self.web.drive({'vx':1.2,'yaw':.5}))
        self.assertAlmostEqual(float(self.web.command[0,0]),.6)
        self.assertAlmostEqual(float(self.web.command[0,2]),.15)
        self.web.drive({'vx':-.55});self.assertEqual(float(self.web.command[0,0]),0.)


if __name__=='__main__':unittest.main()
