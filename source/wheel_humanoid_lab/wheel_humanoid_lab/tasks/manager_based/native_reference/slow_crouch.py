"""CPU-testable timing and physical quality gates for a grounded slow crouch.

The measured video contains sharp depth-estimation changes. Reversing its
timestamps directly would request a 1.5 m/s descent. Retiming keeps every
measured pose in order while bounding reference vertical and joint speeds.
"""
from __future__ import annotations
import torch


class SlowCrouchTrajectory:
    max_height_speed = .16
    max_joint_speed = .8
    max_pitch_speed = .4

    def __init__(self, trajectory):
        self.source = trajectory
        intervals = trajectory.times.diff().flip(0)
        height_time = trajectory.height.diff().abs().flip(0)/self.max_height_speed
        joint_time = trajectory.q.diff(dim=0).abs().amax(-1).flip(0)/self.max_joint_speed
        pitch_time = trajectory.pitch.diff().abs().flip(0)/self.max_pitch_speed
        intervals = torch.stack((intervals, height_time, joint_time, pitch_time)).amax(0)
        self.times = torch.cat((torch.zeros_like(intervals[:1]), intervals.cumsum(0)))
        self.source_phases = ((trajectory.times-trajectory.times[0])/trajectory.duration).flip(0)
        self.duration = float(self.times[-1])

    def source_phase(self, phase):
        time = phase.clamp(0., 1.)*self.duration
        high = torch.searchsorted(self.times, time.contiguous(), right=True).clamp(1, self.times.numel()-1)
        low = high-1
        blend = ((time-self.times[low])/(self.times[high]-self.times[low])).clamp(0., 1.)
        return self.source_phases[low]+blend*(self.source_phases[high]-self.source_phases[low])

    def sample(self, phase):
        return self.source.sample(self.source_phase(phase))

    def sample_pitch(self, phase):
        return self.source.sample_pitch(self.source_phase(phase))


class SlowCrouchQuality:
    """Latch sustained violations; tolerate initial settling and sensor flicker.

    Contact is required at both main wheels, rather than arbitrary hand/calf
    contacts. Velocity is measured in world Z so torso pitch cannot disguise a
    rapid fall. Only the first 0.2 seconds after a real mode change/reset have a
    contact-settling grace; repeated bounces do not renew that grace.
    """
    def __init__(self, num_envs, device):
        self.checked_s = torch.zeros(num_envs, device=device)
        for name in ('bilateral_s', 'bilateral_loss_s', 'max_bilateral_loss_s',
                     'air_s', 'max_air_s', 'fast_descent_s', 'max_fast_descent_s',
                     'hop_s', 'max_hop_s', 'max_descent_mps'):
            setattr(self, name, self.checked_s.clone())
        self.failed = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    @property
    def contact_fraction(self):
        return torch.where(self.checked_s > 0., self.bilateral_s/self.checked_s.clamp_min(1.e-6),
                           torch.ones_like(self.checked_s))

    def update(self, wheel_normal, world_vz, height_error, elapsed, active, dt):
        checked = active & (elapsed >= .2)
        bilateral = (wheel_normal > 8.).all(-1)
        air = (wheel_normal < 5.).all(-1)
        self.checked_s += checked.float()*dt
        self.bilateral_s += (checked & bilateral).float()*dt
        conditions = dict(bilateral_loss=~bilateral, air=air,
                          fast_descent=world_vz < -.38,
                          hop=(world_vz > .24) & (height_error > .03))
        for name, condition in conditions.items():
            current = getattr(self, name+'_s')
            current.copy_(torch.where(checked & condition, current+dt, 0.))
            maximum = getattr(self, 'max_'+name+'_s')
            maximum.copy_(torch.maximum(maximum, current))
        self.max_descent_mps.copy_(torch.maximum(self.max_descent_mps,
                                                torch.where(checked, (-world_vz).clamp_min(0.), 0.)))
        self.failed |= checked & ((self.bilateral_loss_s > .12) | (self.air_s > .06)
                                  | (self.fast_descent_s > .1) | (self.hop_s > .08))
        return bilateral
