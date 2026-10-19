"""Offline CAD FK audit; no claim of dynamic stability or hardware calibration."""
import argparse,ast,json,math,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def rotation(axis,angle):
 a=np.array(axis,dtype=float);a/=np.linalg.norm(a);x,y,z=a;c=math.cos(angle);s=math.sin(angle);d=1-c
 t=np.eye(4);t[:3,:3]=[[c+x*x*d,x*y*d-z*s,x*z*d+y*s],[y*x*d+z*s,c+y*y*d,y*z*d-x*s],[z*x*d-y*s,z*y*d+x*s,c+z*z*d]];return t

def xyzrpy(element):
 t=np.eye(4)
 if element is None:return t
 xyz=[float(v) for v in element.get('xyz','0 0 0').split()];r,p,y=[float(v) for v in element.get('rpy','0 0 0').split()]
 t=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r);t[:3,3]=xyz;return t

def fk(urdf,values):
 joints=urdf.findall('joint');values=dict(values)
 for j in joints:
  m=j.find('mimic')
  if m is not None:values[j.get('name')]=values.get(m.get('joint'),0)*float(m.get('multiplier','1'))+float(m.get('offset','0'))
 parents={j.find('parent').get('link') for j in joints};children={j.find('child').get('link') for j in joints}
 poses={n:np.eye(4) for n in parents-children}
 for _ in joints:
  for j in joints:
   pa=j.find('parent').get('link');ch=j.find('child').get('link')
   if pa not in poses or ch in poses:continue
   a=j.find('axis');axis=[float(v) for v in a.get('xyz','1 0 0').split()] if a is not None else [1,0,0]
   poses[ch]=poses[pa]@xyzrpy(j.find('origin'))@(rotation(axis,values.get(j.get('name'),0)) if j.get('type')!='fixed' else np.eye(4))
 return poses

def current_poses():
 # Evaluate only the existing pure pose function; do not import Kit to inspect CAD.
 source=ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/inertia/native_skills.py'
 tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pose_values')
 assets=ast.parse((ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/assets/q1_inertia.py').read_text())
 names=next(ast.literal_eval(n.value) for n in assets.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='Q1_INERTIA_POSITION_JOINTS' for x in n.targets))
 ns={'Q1_INERTIA_POSITION_JOINTS':names,'Q1_INERTIA_DEFAULT_POS':{n:0. for n in names}}
 # Other nonzero defaults relevant to COM are preserved, too.
 for n in assets.body:
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='Q1_INERTIA_DEFAULT_POS' and n.value.func.attr=='update':exec(compile(ast.Module(body=[n],type_ignores=[]),'defaults','exec'),ns)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
 return {k:ns['pose_values'](k) for k in ('stand','kneel','load','push','tuck')}

def audit():
 path=ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf';urdf=ET.parse(path).getroot();old=ET.parse(ROOT/'urdf/wheel_humanoid.urdf').getroot()
 result={'urdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'angle_units':'radians','frame':'+X forward, +Y left, +Z up; root fixed upright for FK','not_a_dynamic_success_test':True,'poses':{},'axis_comparison':{}}
 for side,short in [('left','l'),('right','r')]:
  result['axis_comparison'][side]={}
  for a,b in [('hip_Pitch','hip_pitch'),('knee_1_Pitch','knee'),('wheel_Pitch','wheel')]:
   new=urdf.find(f"joint[@name='{side}_{a}_joint']");prev=old.find(f"joint[@name='{short}_{b}_joint']")
   result['axis_comparison'][side][a]={'old_axis':prev.find('axis').get('xyz'),'new_axis':new.find('axis').get('xyz'),'old_limit':prev.find('limit').attrib if prev.find('limit') is not None else None,'new_limit':new.find('limit').attrib if new.find('limit') is not None else None}
 for name,q in {'zero':{},**current_poses()}.items():
  poses=fk(urdf,q);feet={side:poses[side+'_wheel_link'][:3,3].tolist() for side in ('left','right')}
  total=0.;com=np.zeros(3)
  for link in urdf.findall('link'):
   inertial=link.find('inertial');mass=float(inertial.find('mass').get('value'));total+=mass;com+=mass*(poses[link.get('name')]@xyzrpy(inertial.find('origin')))[:3,3]
  com/=total;foot=np.mean(list(feet.values()),axis=0)
  root_z=.1-min(v[2] for v in feet.values())
  result['poses'][name]={'q':q,'wheel_centers_root_m':feet,'root_z_for_wheel_bottom_z0_m':root_z,'com_relative_root_m':com.tolist(),'com_minus_wheel_center_x_m':float(com[0]-foot[0]),'configured_reset_root_z_before_fix_m':.45 if name=='kneel' else .848686 if name=='stand' else None,'knees_root_m':{side:(poses[side+'_knee_link']@xyzrpy(urdf.find(f"joint[@name='{side}_knee_1_Pitch_joint']/origin")))[:3,3].tolist() for side in ('left','right')}}
  if name=='kneel':result['poses'][name]['configured_reset_wheel_bottom_z_before_fix_m']=.45+min(v[2] for v in feet.values())-.1
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
 for n,v in r['poses'].items():print(n,'grounded_root_z',round(v['root_z_for_wheel_bottom_z0_m'],4),'COM-foot-x',round(v['com_minus_wheel_center_x_m'],4),'foot_xz',[round(v['wheel_centers_root_m']['left'][i],4) for i in (0,2)])
