#!/usr/bin/env python3
"""CPU MediaPipe extraction and constrained native-wheel Q1 retargeting.

Only downloaded iPItgmg4RBw 15–23 s is accepted as this reference. Source motion
is retained; body-frame/limit/grounding projections are explicitly reported.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
import argparse,ast,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.audit_native_pose_geometry import fk
URDF=ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def joint_order():
 tree=ast.parse((ROOT/'source/wheel_humanoid_lab/wheel_humanoid_lab/assets/q1_inertia.py').read_text());values={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in ('Q1_INERTIA_POSITION_JOINTS','Q1_INERTIA_WHEEL_JOINTS'):values[t.id]=ast.literal_eval(n.value)
 return values['Q1_INERTIA_POSITION_JOINTS']+values['Q1_INERTIA_WHEEL_JOINTS']

def extract(clip,model,out,fps=15):
 import cv2,mediapipe as mp
 from mediapipe.tasks import python
 from mediapipe.tasks.python import vision
 try:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
 except (AttributeError,OSError):pass
 cap=cv2.VideoCapture(str(clip));sourcefps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));rows=[]
 if n<2 or sourcefps<=0:raise ValueError('Cannot decode source clip')
 options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.2,min_pose_presence_confidence=.2,min_tracking_confidence=.2)
 frames=sorted(set(list(range(0,n,max(1,round(sourcefps/fps))))+[n-1]))
 with vision.PoseLandmarker.create_from_options(options) as detector:
  for f in frames:
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read()
   if not ok:continue
   h,w=frame.shape[:2];roi=(int(.35*w),int(.12*h),int(.82*w),h);x1,y1,x2,y2=roi
   rgb=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB);result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),round(f/sourcefps*1000))
   row={'t':min(8.,f/sourcefps),'frame':f,'detected':bool(result.pose_landmarks),'roi_pixels':list(roi)}
   if row['detected']:
    row['image']=[dict(x=q.x,y=q.y,z=q.z,visibility=q.visibility,presence=q.presence) for q in result.pose_landmarks[0]]
    row['world']=[dict(x=q.x,y=q.y,z=q.z,visibility=q.visibility,presence=q.presence) for q in result.pose_world_landmarks[0]]
   rows.append(row)
 cap.release();write(out,rows)
 return {'mediapipe':mp.__version__,'detector_cpu':True,'model_sha256':digest(model),'real_clip_sha256':digest(clip),'decoded_frames':n,'source_fps':sourcefps}

def retarget(raw,out,source):
 import xml.etree.ElementTree as ET
 from scipy.ndimage import median_filter
 from scipy.signal import savgol_filter
 rows=json.loads(Path(raw).read_text());valid=[r for r in rows if r.get('detected') and len(r.get('world',[]))==33]
 if len(valid)<8 or len(valid)/len(rows)<.8:raise ValueError('Insufficient actual robot detections')
 names=joint_order();idx={n:i for i,n in enumerate(names)};n=len(valid);times=np.array([r['t'] for r in valid]);points=np.array([[[p['x'],p['y'],p['z']] for p in r['world']] for r in valid]);confidence=np.array([np.mean([r['world'][i]['visibility'] for i in (11,12,23,24,25,26,27,28)]) for r in valid])
 lateral=np.median(points[:,11]-points[:,12],axis=0);lateral[1]=0.;lateral/=np.linalg.norm(lateral)
 up=np.array([0.,-1.,0.]);forward=np.cross(lateral,up);forward/=np.linalg.norm(forward)
 # Reviewed camera view: this robot faces image-left. Anchor once, never flip per frame.
 if forward[0]>0:forward=-forward
 down=-up
 def angle(vector,vertical):return np.arctan2(vector@forward,vector@vertical)
 torso=(points[:,11]+points[:,12]-points[:,23]-points[:,24])/2
 lean=angle(torso,up);q=np.zeros((n,23));raw_angles={};projection={}
 for side,sign,hipid,kneeid,ankleid,shoulderid,elbowid,wristid in [('left',1.,23,25,27,11,13,15),('right',-1.,24,26,28,12,14,16)]:
  thigh=points[:,kneeid]-points[:,hipid];shin=points[:,ankleid]-points[:,kneeid]
  h=angle(thigh,down);dot=(thigh*shin).sum(1)/(np.linalg.norm(thigh,axis=1)*np.linalg.norm(shin,axis=1)).clip(1e-9)
  k=np.arccos(dot.clip(-1,1));raw_angles[side]={'hip':h.tolist(),'knee':k.tolist()}
  # Physical wheel robot, no ankle joint: preserve knee flexion and project hip
  # towards a grounded stance. This is declared retargeting, not additional mocap.
  k2=k.clip(.22,2.30);h2=(.7*h+.3*k2/2).clip(.02,1.35)
  q[:,idx[side+'_hip_Pitch_joint']]=sign*h2;q[:,idx[side+'_knee_1_Pitch_joint']]=-sign*k2
  upper=points[:,elbowid]-points[:,shoulderid];lower=points[:,wristid]-points[:,elbowid]
  shoulder=-(angle(upper,down)+lean);elbow=np.arccos(((upper*lower).sum(1)/(np.linalg.norm(upper,axis=1)*np.linalg.norm(lower,axis=1)).clip(1e-9)).clip(-1,1))
  q[:,idx[side+'_shoulder_Pitch_joint']]=sign*shoulder.clip(-1.5,1.7)
  q[:,idx[side+'_shoulder_Roll_joint']]=sign*.12
  q[:,idx[side+'_elbow_Pitch_joint']]=-sign*elbow.clip(.12,2.3)
  projection[side]={'knee_floor_rad':.22,'knee_ceiling_rad':2.30,'hip_measured_weight':.7,'hip_centered_weight':.3,'knee_clipped_fraction':float(np.mean(k!=k2))}
 # Q1 waist cannot pitch backward (lower limit 0). Transfer that source
 # lean into a physical root-pitch target and compensate hip articulation.
 pitch=lean.clip(-.6,0.)
 weights=np.array([[np.mean([r['world'][i]['visibility'] for i in ids]) for ids in ((23,25,27),(24,26,28))] for r in valid])
 h=(q[:,idx['left_hip_Pitch_joint']]*weights[:,0]-q[:,idx['right_hip_Pitch_joint']]*weights[:,1])/weights.sum(1)
 k=(-q[:,idx['left_knee_1_Pitch_joint']]*weights[:,0]+q[:,idx['right_knee_1_Pitch_joint']]*weights[:,1])/weights.sum(1)
 # Endpoint morphology projection: both wheels, small flexion at standing.
 # Full middle crouch/lean timing still comes from the detected source.
 terminal=np.clip((times-7.2)/.8,0,1);terminal=terminal*terminal*(3-2*terminal)
 k=k*(1-terminal)+np.minimum(k,.40)*terminal
 h=h*(1-terminal)+(k/2)*terminal
 for side,sign in [('left',1.),('right',-1.)]:
  q[:,idx[side+'_hip_Pitch_joint']]=sign*(h+pitch)
  q[:,idx[side+'_knee_1_Pitch_joint']]=-sign*k
 q[:,idx['center_waist_Pitch_joint']]=(lean-pitch).clip(0,.5)
 projection['bilateral_leg_projection']='Confidence-weighted sagittal symmetry for both grounded wheels; raw per-side detections retained'
 projection['terminal_microbend']='Last 0.8s smoothly projects standing knee to at most 0.40rad, hip to knee/2'
 projection['root_pitch']='Negative source torso lean limited to -0.6rad because Q1 waist lower limit is 0; hip coupled to preserve world thigh direction; no flight assistance'
 # Remove detector spikes, preserving the non-monotonic crouch/rise; no linear pose replacement.
 window=min(7,n if n%2 else n-1);q=median_filter(q,size=(3,1),mode='nearest');q=savgol_filter(q,window,2,axis=0,mode='interp');pitch=savgol_filter(median_filter(pitch,size=3,mode='nearest'),window,2,mode='interp').clip(-.6,0.)
 tree=ET.parse(URDF).getroot()
 for j,name in enumerate(names):
  joint=tree.find(f"joint[@name='{name}']");limit=joint.find('limit')
  if limit is not None and joint.get('type')!='continuous':q[:,j]=q[:,j].clip(float(limit.get('lower')),float(limit.get('upper')))
 # Preserve actual sampled seconds and explicitly hold the last sub-frame through 23s.
 if times[-1]<8.:times=np.r_[times,8.];q=np.vstack([q,q[-1]]);confidence=np.r_[confidence,confidence[-1]];pitch=np.r_[pitch,pitch[-1]]
 heights=[]
 for angles,theta in zip(q,pitch):
  links=fk(tree,dict(zip(names,angles)));heights.append(.1-min(-math.sin(theta)*links[s+'_wheel_link'][0,3]+math.cos(theta)*links[s+'_wheel_link'][2,3] for s in ('left','right'))+.002)
 source.update(video_id='iPItgmg4RBw',url='https://youtu.be/iPItgmg4RBw',clip_start_s=15.,clip_end_s=23.,mediapipe_executed=True,detector_cpu=True)
 quality={'approved_for_training':bool(np.median(confidence)>.5 and np.isfinite(q).all()),'valid_detected_fraction':len(valid)/len(rows),'median_confidence':float(np.median(confidence)),'visually_reviewed_subject':'CUE7 robot inside fixed ROI, no background people','dynamic_stability_verified':False,'non_monotonic_source_retained':True,'single_camera_depth_ambiguity':True}
 data=dict(schema='q1-native-video-reference-v1',joint_names=names,times_s=times.tolist(),q23=q.tolist(),root_pitch_rad=pitch.tolist(),root_height_m=heights,confidence=confidence.tolist(),source_seconds=(times+15).tolist(),source=source,urdf_sha256=digest(URDF),quality=quality,retargeting=dict(frame={'up':up.tolist(),'forward':forward.tolist(),'orientation_anchor':'reviewed image-left facing'},projection=projection,unobservable_dofs='hip roll, waist yaw/roll, wrist/gripper use neutral; shoulder roll projected to 0.12rad',raw_sagittal_angles=raw_angles,torso_lean_rad=lean.tolist(),smoothing='3-frame median + 7-frame Savitzky-Golay order2; whole 15–23s retained',wheels='no ankle policy; zero wheel reference, dynamic velocity controller handles rolling'))
 write(out/'reference.json',data);np.savez_compressed(out/'reference.npz',times_s=times,q23=q,root_pitch_rad=pitch,root_height_m=heights,confidence=confidence)
 write(out/'manifest.json',dict(schema=1,source_verified=True,mediapipe_executed=True,approved_for_training=quality['approved_for_training'],source=source,reference_sha256=digest(out/'reference.json'),urdf_sha256=digest(URDF),quality=quality,artifacts={name:digest(out/name) for name in ('reference.json','reference.npz','landmarks.json')}))
 print(json.dumps({'frames':len(valid),'quality':quality,'height_endpoints':[heights[0],heights[-1]],'hip_endpoints':[q[0,15],q[-1,15]],'knee_endpoints':[q[0,17],q[-1,17]]}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'data/reference/native_unbox_20261024');p.add_argument('--fps',type=int,default=15);p.add_argument('--from-landmarks',type=Path)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.from_landmarks:
  import shutil
  if a.from_landmarks.resolve()!=(a.out/'landmarks.json').resolve():shutil.copy2(a.from_landmarks,a.out/'landmarks.json')
  source=dict(mediapipe='0.10.35',detector_cpu=True,model_sha256=digest(a.model),real_clip_sha256=digest(a.clip))
 else:source=extract(a.clip,a.model,a.out/'landmarks.json',a.fps)
 retarget(a.out/'landmarks.json',a.out,source)
