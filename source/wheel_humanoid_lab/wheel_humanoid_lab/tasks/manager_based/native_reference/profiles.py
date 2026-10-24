"""Tensor-only motor profiles; no simulator state mutation or flight assistance."""
import math
import torch


def smooth(t):
    t=t.clamp(0.,1.);return t*t*t*(10-15*t+6*t*t)


def general_leg_angles(x, down):
    c=((x*x+down*down-.32**2-.34**2)/(2*.32*.34)).clamp(-.9999,.9999)
    knee=torch.acos(c)
    hip=torch.atan2(x,down)+torch.atan2(.34*torch.sin(knee),.32+.34*torch.cos(knee))
    return hip,knee


def backflip_reference(t, stand, crouch, has_air, takeoff_t, winding, landed, touchdown_t, indices, level=1.):
    """Calibrated launch sequence; the unbox video is not called flip mocap.

    Old wheel-Q1's successful arm-swing/single-load ordering informs this profile,
    but no old 24-action checkpoint or joint angles bypass native limits. Actual
    rising flight/contact controls tuck and landing; timing alone cannot certify a flip.
    """
    load=crouch.clone();push=stand.clone();tuck=stand.clone()
    for side,sign in (('left',1.),('right',-1.)):
        load[:,indices[side+'_shoulder_Pitch_joint']]=.60*sign
        push[:,indices[side+'_hip_Pitch_joint']]=-.45*sign
        push[:,indices[side+'_knee_1_Pitch_joint']]=-.10*sign
        push[:,indices[side+'_shoulder_Pitch_joint']]=-1.00*sign
        push[:,indices[side+'_elbow_Pitch_joint']]=-.30*sign
        tuck[:,indices[side+'_hip_Pitch_joint']]=-.85*sign
        tuck[:,indices[side+'_knee_1_Pitch_joint']]=-2.25*sign
        tuck[:,indices[side+'_shoulder_Pitch_joint']]=-.80*sign
        tuck[:,indices[side+'_elbow_Pitch_joint']]=-1.4*sign
    load[:,indices['center_waist_Pitch_joint']]=.24
    push[:,indices['center_waist_Pitch_joint']]=.04
    tuck[:,indices['center_waist_Pitch_joint']]=.15
    ref=stand+(load-stand)*smooth((t-.5)/.7)[:,None]
    ref=torch.where((t>=1.2)[:,None],load+(push-load)*smooth((t-1.2)/.18)[:,None],ref)
    flight=push+(tuck-push)*smooth((t-takeoff_t-.04)/.14)[:,None]
    ref=torch.where(has_air[:,None],flight,ref)
    opening=(-winding>4.7)&has_air
    ref=torch.where(opening[:,None],tuck+(stand-tuck)*smooth((-winding-4.7)/1.1)[:,None],ref)
    landing=ref+(stand-ref)*smooth((t-touchdown_t)/.7)[:,None]
    ref=torch.where(landed[:,None],landing,ref)
    # Early curriculum reduces amplitude, not gravity or mass. Final eval is 1.
    if isinstance(level,torch.Tensor):level=level[:,None] if level.ndim else level
    return stand+(ref-stand)*level
