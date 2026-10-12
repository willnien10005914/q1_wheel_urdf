"""Keyboard routing regressions: commands must reach the standing policy."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from skills.teleop import KeyState,IsaacKeyboard,apply_drive
class Matrix:
    def __init__(self):self.values=[0.,0.,0.]
    def __setitem__(self,key,value):self.values[key[1]]=value
class Robot:
    def __init__(self,mode='stand'):
        self.mode=mode;self.command=Matrix();self.requests=[]
    def request(self,name):self.requests.append(name);self.mode=name;return True
class ControlsTest(unittest.TestCase):
    def test_isaac_press_and_release_reach_drive_command(self):
        k=IsaacKeyboard.__new__(IsaacKeyboard);KeyState.__init__(k)
        k.carb=SimpleNamespace(KeyboardEventType=SimpleNamespace(KEY_PRESS='press',KEY_RELEASE='release'))
        robot=Robot()
        k._on_event(SimpleNamespace(type='press',input=SimpleNamespace(name='W')))
        self.assertTrue(apply_drive(robot,k.command_update()))
        self.assertEqual(robot.requests,['skate']);self.assertEqual(robot.command.values,[1.2,0.,0.])
        k._on_event(SimpleNamespace(type='release',input=SimpleNamespace(name='W')))
        apply_drive(robot,k.command_update());self.assertEqual(robot.command.values,[0.,0.,0.])
    def test_turn_stop_and_pose_keys(self):
        k=KeyState();k.event('W',True);k.event('A',True)
        self.assertEqual(k.command_update(),{'vx':1.2,'vy':0.,'yaw':.45})
        k.event('SPACE',True);self.assertEqual(k.command_update()['vx'],0.)
        k.event('P',True);self.assertEqual(k.pop_pose('skate'),'slide')
        k.event('R',True);self.assertEqual(k.pop_pose('slide'),'reset')
    def test_drive_does_not_override_backflip(self):
        robot=Robot('backflip');self.assertFalse(apply_drive(robot,{'vx':1.2}))
        self.assertEqual(robot.mode,'backflip');self.assertEqual(robot.requests,[])
    def test_drive_sets_target_without_jumping_filtered_command(self):
        robot=Robot();robot.command_target=Matrix()
        apply_drive(robot,{'vx':1.2})
        self.assertEqual(robot.command_target.values,[1.2,0.,0.])
        self.assertEqual(robot.command.values,[0.,0.,0.])
    def test_space_pause_is_resumed_on_kit_update(self):
        k=IsaacKeyboard.__new__(IsaacKeyboard)
        class Timeline:
            resumed=False
            def is_stopped(self):return False
            def is_playing(self):return self.resumed
            def play(self):self.resumed=True
        k.timeline=Timeline();k._keep_playing(None)
        self.assertTrue(k.timeline.resumed)
    def test_invalid_command_is_ignored(self):
        robot=Robot();self.assertFalse(apply_drive(robot,{'vx':float('nan')}))
        self.assertEqual(robot.requests,[])
if __name__=='__main__':unittest.main()
