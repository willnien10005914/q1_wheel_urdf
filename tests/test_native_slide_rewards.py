"""Reward loopholes and CAD leg kinematics, without a simulator."""
import importlib.util
from pathlib import Path
import unittest
import math
import xml.etree.ElementTree as ET
import torch

path = Path(__file__).resolve().parents[1]/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/native_slide/gait.py'
spec = importlib.util.spec_from_file_location('gait', path)
gait = importlib.util.module_from_spec(spec); spec.loader.exec_module(gait)


class SlideRewardsTest(unittest.TestCase):
    def test_push_requires_ground_backward_motion_and_forward_force(self):
        stance=torch.tensor([[True,False]]*5)
        load=torch.tensor([[200.,0.],[0.,0.],[200.,0.],[200.,0.],[200.,0.]])
        vx=torch.tensor([[-.2,.3],[-.2,.3],[.2,.3],[-.2,.3],[-.2,.3]])
        force=torch.tensor([[20.,0.],[20.,0.],[20.,0.],[-20.,0.],[20.,0.]])
        forward=torch.tensor([.4,.4,.4,.4,0.])
        push,_,_=gait.measured_rewards(stance,load,force,vx,torch.full((5,2),.035),forward)
        self.assertGreater(float(push[0]),0.)
        torch.testing.assert_close(push[1:],torch.zeros(4))

    def test_airborne_both_feet_does_not_earn_recovery(self):
        push,recovery,penalty=gait.measured_rewards(torch.tensor([[False,True]]),torch.zeros(1,2),
            torch.zeros(1,2),torch.full((1,2),.3),torch.full((1,2),.035),torch.tensor([.4]))
        self.assertEqual(float(push+recovery),0.)
        self.assertGreaterEqual(float(penalty),1.)

    def test_lifting_without_loaded_push_never_completes_a_stroke(self):
        trackers=[gait.CycleTracker(1,'cpu') for _ in range(3)]
        for step in range(500):
            _,stance,_,lift=gait.profile(torch.tensor([step*.02]))
            normal=torch.where(stance | (lift<.008),200.,0.)
            for i,tracker in enumerate(trackers):
                v=torch.where(stance,-.25 if i else 0.,.4)
                contacts=normal.clone()
                if i==2 and bool((normal<5.).any()):contacts.zero_()
                tracker.update(stance,contacts,v,torch.where(stance,20.,0.),torch.tensor([.4]),.02)
        self.assertEqual(int(trackers[0].cycles.sum()),0)
        self.assertTrue(bool((trackers[1].cycles>=3).all()))
        self.assertEqual(int(trackers[2].cycles.sum()),0)

    def test_cad_ik_retains_ground_height_during_support(self):
        _,stance,x,lift=gait.profile(torch.linspace(0,1.4,141))
        hip,knee=gait.leg_angles(x,lift)
        fk_x=.32*torch.sin(hip)+.34*torch.sin(hip-knee)
        down=.32*torch.cos(hip)+.34*torch.cos(hip-knee)
        torch.testing.assert_close(fk_x,x-.02*torch.sin(torch.tensor(.35)),atol=1.e-6,rtol=1.e-5)
        torch.testing.assert_close(down,.66*torch.cos(torch.tensor(.35))-lift,atol=1.e-6,rtol=1.e-5)
        self.assertTrue(bool((lift[stance]==0).all()))
        self.assertLess(float(lift.max()),.036)


    def test_reference_matches_actual_urdf_leg_chain(self):
        urdf=ET.parse(Path(__file__).resolve().parents[1]/'urdf/q1_inertia/quanta_robot_wheeled.urdf').getroot()
        joints=urdf.findall('joint')
        def rotation(axis,t):
            x,y,z=axis;c=math.cos(t);sn=math.sin(t);d=1-c
            return torch.tensor([[c+x*x*d,x*y*d-z*sn,x*z*d+y*sn,0],
                [y*x*d+z*sn,c+y*y*d,y*z*d-x*sn,0],
                [z*x*d-y*sn,z*y*d+x*sn,c+z*z*d,0],[0,0,0,1]],dtype=torch.float64)
        for t in [0.,.25,.55,.85,1.1]:
            _,_,x,lift=gait.profile(torch.tensor([t]));hip,knee=gait.leg_angles(x,lift)
            q={}
            for i,side in enumerate(('left','right')):
                sign=1 if i==0 else -1
                q[side+'_hip_Pitch_joint']=sign*float(hip[0,i]);q[side+'_knee_1_Pitch_joint']=-sign*float(knee[0,i])
            for joint in joints:
                m=joint.find('mimic')
                if m is not None:q[joint.get('name')]=q.get(m.get('joint'),0)*float(m.get('multiplier','1'))
            parents={j.find('parent').get('link') for j in joints};children={j.find('child').get('link') for j in joints}
            poses={name:torch.eye(4,dtype=torch.float64) for name in parents-children}
            for _ in joints:
                for j in joints:
                    pa=j.find('parent').get('link');ch=j.find('child').get('link')
                    if pa not in poses or ch in poses:continue
                    o=j.find('origin');rpy=list(map(float,o.get('rpy','0 0 0').split()));xyz=list(map(float,o.get('xyz','0 0 0').split()))
                    transform=rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0]);transform[:3,3]=torch.tensor(xyz)
                    axis=j.find('axis');axis=list(map(float,axis.get('xyz').split())) if axis is not None else [1,0,0]
                    poses[ch]=poses[pa]@transform@rotation(axis,q.get(j.get('name'),0))
            for i,side in enumerate(('left','right')):
                foot=poses[side+'_wheel_link'][:3,3]
                self.assertAlmostEqual(float(foot[0]),float(x[0,i])-.02*math.sin(.35),places=5)
                self.assertAlmostEqual(float(foot[2]),-.1267-.66*math.cos(.35)+float(lift[0,i]),places=5)

    def test_forward_travel_is_not_mistaken_for_a_fall(self):
        failed=gait.failure_mask(torch.tensor([.83,.83,.2,.83]),torch.ones(4),torch.zeros(4,dtype=torch.bool),torch.tensor([[8.,0.],[12.1,0.],[0.,0.],[0.,0.]]))
        self.assertEqual(failed.tolist(),[False,True,True,False])


if __name__=='__main__':unittest.main()
