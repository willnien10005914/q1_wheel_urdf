"""Reset placement must match full CAD FK, including knee mirror/mimic signs."""
import ast,math
from functools import lru_cache
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from tools.audit_native_pose_geometry import current_poses,fk
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/inertia/native_skills.py'
fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='reset_root_height')
poses=current_poses();ns=dict(math=math,ET=ET,lru_cache=lru_cache,Q1_INERTIA_URDF=str(ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf'),Q1_INERTIA_WHEEL_RADIUS=.1,pose_values=poses.__getitem__)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)

class ResetGeometryTest(unittest.TestCase):
    def test_both_wheels_have_2mm_clearance_at_stand_and_kneel_reset(self):
        urdf=ET.parse(ns['Q1_INERTIA_URDF']).getroot()
        for kind in ('stand','kneel'):
            links=fk(urdf,poses[kind]);height=ns['reset_root_height'](kind)
            for side in ('left','right'):
                self.assertAlmostEqual(height+links[side+'_wheel_link'][2,3]-.1,.002,places=8)

    def test_stand_reset_retains_existing_height(self):
        self.assertAlmostEqual(ns['reset_root_height']('stand'),.8486859904792701,places=8)

    def test_kneel_reset_does_not_use_standing_angle_or_buried_wheels(self):
        height=ns['reset_root_height']('kneel')
        self.assertGreater(height,.473);self.assertLess(height,.475)

if __name__=='__main__':unittest.main()
