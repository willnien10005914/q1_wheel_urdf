"""Physical crouch gates and timing, without Isaac or a GPU."""
import importlib.util
import json
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/native_reference'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

slow=load('reference_slow_crouch',TASK/'slow_crouch.py')
trajectory=load('reference_crouch_source',TASK/'trajectory.py')


class SlowTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'data/reference/native_unbox_20261024/reference.json'
        names=json.loads(path.read_text())['joint_names']
        cls.source=trajectory.VideoTrajectory(path,names,ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf')
        cls.crouch=slow.SlowCrouchTrajectory(cls.source)

    def test_keeps_every_measured_pose_in_reverse_order(self):
        q,h,c=self.crouch.sample(self.crouch.times/self.crouch.duration)
        torch.testing.assert_close(q,self.source.q.flip(0),atol=5.e-6,rtol=5.e-6)
        torch.testing.assert_close(h,self.source.height.flip(0),atol=5.e-6,rtol=5.e-6)
        self.assertGreater(self.crouch.duration,self.source.duration)
        self.assertLess(self.crouch.duration+4.,36.)

    def test_all_segment_vertical_joint_and_pitch_speeds_are_bounded(self):
        dt=self.crouch.times.diff()
        q=self.source.q.flip(0);h=self.source.height.flip(0);p=self.source.pitch.flip(0)
        self.assertLessEqual(float((h.diff().abs()/dt).max()),self.crouch.max_height_speed+1.e-5)
        self.assertLessEqual(float((q.diff(dim=0).abs()/dt[:,None]).max()),self.crouch.max_joint_speed+1.e-5)
        self.assertLessEqual(float((p.diff().abs()/dt).max()),self.crouch.max_pitch_speed+1.e-5)

    def test_endpoint_holds_do_not_extrapolate(self):
        q,h,_=self.crouch.sample(torch.tensor([-1.,0.,1.,2.]))
        torch.testing.assert_close(q[0],self.source.q[-1]);torch.testing.assert_close(q[-1],self.source.q[0])
        torch.testing.assert_close(h[0],self.source.height[-1]);torch.testing.assert_close(h[-1],self.source.height[0])


class SlowQualityTest(unittest.TestCase):
    def setUp(self):self.s=slow.SlowCrouchQuality(2,'cpu')
    def step(self,wheels=((100.,100.),(100.,100.)),vz=(0.,0.),error=(0.,0.),elapsed=.3,dt=.02):
        self.s.update(torch.tensor(wheels),torch.tensor(vz),torch.tensor(error),
                      torch.full((2,),elapsed),torch.ones(2,dtype=torch.bool),dt)

    def test_grounded_slow_descent_passes(self):
        for _ in range(200):self.step(vz=(-.16,-.12))
        self.assertFalse(bool(self.s.failed.any()))
        torch.testing.assert_close(self.s.contact_fraction,torch.ones(2))

    def test_transient_contact_noise_recovers_without_latched_failure(self):
        for _ in range(100):self.step()
        for _ in range(2):self.step(wheels=((0.,0.),(0.,100.)))
        for _ in range(100):self.step()
        self.assertFalse(bool(self.s.failed.any()))
        self.assertTrue(bool((self.s.contact_fraction>.95).all()))

    def test_true_airborne_crouch_fails_even_if_other_bodies_touch_ground(self):
        for _ in range(4):self.step(wheels=((0.,0.),(100.,100.)))
        self.assertTrue(bool(self.s.failed[0]));self.assertFalse(bool(self.s.failed[1]))
        self.step();self.assertTrue(bool(self.s.failed[0]))

    def test_sustained_single_wheel_support_fails(self):
        for _ in range(8):self.step(wheels=((0.,100.),(100.,100.)))
        self.assertTrue(bool(self.s.failed[0]));self.assertFalse(bool(self.s.failed[1]))

    def test_fast_fall_and_upward_prejump_fail_independently(self):
        for _ in range(7):self.step(vz=(-.5,.4),error=(0.,.05))
        self.assertTrue(bool(self.s.failed.all()))

    def test_only_initial_mode_settling_has_a_grace(self):
        for _ in range(20):self.step(wheels=((0.,0.),(0.,0.)),elapsed=.1)
        self.assertFalse(bool(self.s.failed.any()))
        self.assertEqual(float(self.s.checked_s.sum()),0.)
        for _ in range(4):self.step(wheels=((0.,0.),(0.,0.)),elapsed=.3)
        self.assertTrue(bool(self.s.failed.all()))

    def test_repeated_contact_flicker_fails_terminal_contact_fraction_gate(self):
        for _ in range(20):
            self.step(wheels=((0.,100.),(100.,100.)))
            for _ in range(5):self.step()
        self.assertFalse(bool(self.s.failed.any()))
        self.assertLess(float(self.s.contact_fraction[0]),.95)
        self.assertEqual(float(self.s.contact_fraction[1]),1.)

    def test_reset_clears_only_requested_environment(self):
        for _ in range(4):self.step(wheels=((0.,0.),(0.,0.)))
        self.s.reset(torch.tensor([0]))
        self.assertFalse(bool(self.s.failed[0]));self.assertTrue(bool(self.s.failed[1]))
        self.assertEqual(float(self.s.max_air_s[0]),0.)


if __name__=='__main__':unittest.main()
