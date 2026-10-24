"""Strict video reference loading and interpolation, usable without Isaac Sim.

A compatible tensor shape does not make a legacy policy a video policy. Source,
MediaPipe provenance, URDF hash and the exact ordered independent joints are
part of this task's contract.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import torch

CONTRACT = 'q1-native-video-v1-23x90'
SCHEMA = 'q1-native-video-reference-v1'
VIDEO_ID = 'iPItgmg4RBw'
RELATIVE_REFERENCE = 'data/reference/native_unbox_20261024/reference.json'


def sample_linear(times, values, phase):
    """Piecewise linear interpolation, with exact clamped endpoint holds."""
    t = phase.clamp(0., 1.)*(times[-1]-times[0])+times[0]
    high = torch.searchsorted(times, t.contiguous(), right=True).clamp(1, times.numel()-1)
    low = high-1
    fraction = ((t-times[low])/(times[high]-times[low])).clamp(0., 1.)
    while fraction.ndim < values[low].ndim: fraction = fraction.unsqueeze(-1)
    return values[low]+fraction*(values[high]-values[low])


def _rotation(axis, angle):
    xyz = torch.tensor(axis, device=angle.device, dtype=angle.dtype)
    xyz = xyz/xyz.norm().clamp_min(1.e-9)
    x,y,z = xyz; c=angle.cos();s=angle.sin();d=1-c
    result = torch.zeros(*angle.shape, 4, 4, device=angle.device, dtype=angle.dtype)
    result[...,0,0]=c+x*x*d;result[...,0,1]=x*y*d-z*s;result[...,0,2]=x*z*d+y*s
    result[...,1,0]=y*x*d+z*s;result[...,1,1]=c+y*y*d;result[...,1,2]=y*z*d-x*s
    result[...,2,0]=z*x*d-y*s;result[...,2,1]=z*y*d+x*s;result[...,2,2]=c+z*z*d
    result[...,3,3]=1.;return result


def fk_wheel_heights(urdf_path, joint_names, positions, radius=.1, clearance=.002, root_pitch=None):
    """Full source-chain FK, including mimic and source joint origin rotations."""
    urdf=ET.parse(urdf_path).getroot(); joints=urdf.findall('joint')
    q={name:positions[...,i] for i,name in enumerate(joint_names)}
    zeros=torch.zeros(positions.shape[:-1],device=positions.device,dtype=positions.dtype)
    for joint in joints:
        mimic=joint.find('mimic')
        if mimic is not None:
            q[joint.get('name')]=q.get(mimic.get('joint'),zeros)*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
    parents={j.find('parent').get('link') for j in joints};children={j.find('child').get('link') for j in joints}
    identity=torch.eye(4,device=positions.device,dtype=positions.dtype).expand(*positions.shape[:-1],4,4)
    root_transform=identity if root_pitch is None else _rotation([0.,1.,0.],root_pitch)
    transforms={name:root_transform for name in parents-children};pending=list(joints)
    while pending:
        remaining=[]
        for joint in pending:
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in transforms:remaining.append(joint);continue
            origin=joint.find('origin');xyz=[float(v) for v in origin.get('xyz','0 0 0').split()] if origin is not None else [0.,0.,0.]
            rpy=[float(v) for v in origin.get('rpy','0 0 0').split()] if origin is not None else [0.,0.,0.]
            constant=identity.clone()
            for axis,angle in [([0.,0.,1.],rpy[2]),([0.,1.,0.],rpy[1]),([1.,0.,0.],rpy[0])]:
                constant=constant@_rotation(axis,zeros+angle)
            constant[...,:3,3]=torch.tensor(xyz,device=positions.device,dtype=positions.dtype)
            axis=joint.find('axis');axis=[float(v) for v in axis.get('xyz','1 0 0').split()] if axis is not None else [1.,0.,0.]
            movement=identity if joint.get('type')=='fixed' else _rotation(axis,q.get(joint.get('name'),zeros))
            transforms[child]=transforms[parent]@constant@movement
        if len(remaining)==len(pending):raise ValueError('URDF joint tree is disconnected or cyclic')
        pending=remaining
    feet=torch.stack([transforms[s+'_wheel_link'][...,2,3] for s in ('left','right')],dim=-1)
    return radius-feet.min(-1).values+clearance


class VideoTrajectory:
    def __init__(self, path, joint_names, urdf_path, device='cpu'):
        self.path=Path(path).resolve();self.urdf_path=Path(urdf_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'MediaPipe video reference missing: {self.path}; deferred campaign must not substitute hand-authored motion')
        raw=self.path.read_bytes();data=json.loads(raw)
        if data.get('schema')!=SCHEMA:raise ValueError('Unknown reference schema')
        if data.get('joint_names')!=list(joint_names):raise ValueError('Reference joint order is incompatible with native Q1')
        source=data.get('source',{})
        if source.get('video_id')!=VIDEO_ID or float(source.get('clip_start_s',-1))!=15. or float(source.get('clip_end_s',-1))!=23.:
            raise ValueError('Reference must come from requested iPItgmg4RBw 15–23 seconds')
        if not source.get('mediapipe') or not source.get('detector_cpu') or len(str(source.get('model_sha256','')))!=64:
            raise ValueError('Reference lacks real CPU MediaPipe provenance')
        self.sha256=hashlib.sha256(raw).hexdigest();self.urdf_sha256=hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
        declared=data.get('urdf_sha256',source.get('urdf_sha256'))
        if declared and declared!=self.urdf_sha256:raise ValueError('Reference was retargeted for a different URDF')
        self.joint_names=list(joint_names);self.source=source;self.quality=data.get('quality',{})
        if self.quality.get('approved_for_training') is not True or source.get('mediapipe_executed') is not True:raise ValueError('Reference quality approval missing')
        self.times=torch.as_tensor(data['times_s'],dtype=torch.float32,device=device)
        self.q=torch.as_tensor(data['q23'],dtype=torch.float32,device=device)
        self.height=torch.as_tensor(data['root_height_m'],dtype=torch.float32,device=device)
        self.pitch=torch.as_tensor(data.get('root_pitch_rad',[0.]*len(data['times_s'])),dtype=torch.float32,device=device)
        self.confidence=torch.as_tensor(data['confidence'],dtype=torch.float32,device=device)
        n=self.times.numel()
        if n<8 or self.q.shape!=(n,23) or self.height.shape!=(n,) or self.confidence.shape!=(n,):raise ValueError('Malformed reference arrays')
        if any(not bool(torch.isfinite(v).all()) for v in (self.times,self.q,self.height,self.confidence)):
            raise ValueError('Nonfinite reference data')
        if not bool((self.times[1:]>self.times[:-1]).all()) or abs(float(self.times[0]))>1.e-5 or float(self.times[-1])<.5:
            raise ValueError('Reference times must start at zero and strictly increase')
        if not bool(((self.confidence>=0)&(self.confidence<=1)).all()) or float(self.confidence.median())<.2:
            raise ValueError('MediaPipe confidence is too low to train')
        if float(self.quality.get('valid_detected_fraction',1.))<.25:
            raise ValueError('Too few actual detected frames; do not train on invented interpolation')
        urdf=ET.parse(self.urdf_path).getroot()
        for i,name in enumerate(joint_names):
            joint=urdf.find(f"joint[@name='{name}']")
            if joint is None:raise ValueError(f'Unknown reference joint {name}')
            limit=joint.find('limit')
            if joint.get('type')!='continuous' and limit is not None:
                lo=float(limit.get('lower','-inf'));hi=float(limit.get('upper','inf'))
                if not bool(((self.q[:,i]>=lo-1.e-5)&(self.q[:,i]<=hi+1.e-5)).all()):raise ValueError(f'Reference exceeds CAD limits: {name}')
        computed=fk_wheel_heights(self.urdf_path,joint_names,self.q,root_pitch=self.pitch)
        if float((computed-self.height).abs().max())>.015:
            raise ValueError('Reference root height disagrees with source CAD FK; correct grounding before training')
        self.duration=float(self.times[-1]-self.times[0])
        if self.pitch.shape!=(n,) or not torch.isfinite(self.pitch).all() or self.pitch.abs().max()>.65:raise ValueError('Invalid root pitch reference')

    def sample(self, phase, reverse=False):
        phase=1-phase if reverse else phase
        return (sample_linear(self.times,self.q,phase),sample_linear(self.times,self.height,phase),
                sample_linear(self.times,self.confidence,phase))

    def sample_pitch(self,phase,reverse=False):
        return sample_linear(self.times,self.pitch,1-phase if reverse else phase)
