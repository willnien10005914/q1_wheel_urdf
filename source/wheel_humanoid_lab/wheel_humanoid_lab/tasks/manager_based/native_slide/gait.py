"""Measured skating objectives; no simulator imports or state writes."""
import math
import torch

CONTRACT = 'q1-native-slide-v1-23x90'
PERIOD = 1.4
STANCE = .66


def profile(t):
    phase = (t[:, None]/PERIOD + torch.tensor([0., .5], device=t.device)) % 1.
    stance = phase < STANCE
    u = torch.where(stance, phase/STANCE, (phase-STANCE)/(1-STANCE)).clamp(0, 1)
    smooth = u*u*u*(10-15*u+6*u*u)
    x = torch.where(stance, .09*(1-2*smooth), .09*(-1+2*smooth))
    lift = torch.where(stance, 0., .035*torch.sin(math.pi*u).square())
    return phase, stance, x, lift


def leg_angles(x, lift):
    # Exact sagittal FK of the checked-in CAD leg: L1=.32, L2=.34.
    x = x-.02*math.sin(.35)
    down = .66*math.cos(.35)-lift
    c = ((x*x+down*down-.32**2-.34**2)/(2*.32*.34)).clamp(-.9999, .9999)
    knee = torch.acos(c)
    hip = torch.atan2(x, down)+torch.atan2(.34*torch.sin(knee), .32+.34*torch.cos(knee))
    return hip, knee


def measured_rewards(stance, normal, force_x, relative_vx, clearance, body_vx):
    loaded = normal > 60.
    moving = (body_vx > .10)[:, None]
    # Backward motion is relative to the torso. GRF must push the robot forward.
    push = stance*loaded*moving*(-relative_vx/.25).clamp(0, 1)*(force_x/20.).clamp(0, 1)
    recovery = (~stance)*(normal < 5.)*loaded.flip(-1)*moving
    recovery = recovery*(relative_vx/.30).clamp(0, 1)*torch.exp(-((clearance-.035)/.02).square())
    recovery *= (clearance > .012)&(clearance < .07)
    both_air = (normal < 5.).all(-1).float()
    high_lift = (clearance-.07).clamp_min(0).sum(-1)/.035
    return push.mean(-1), recovery.mean(-1), both_air+high_lift


class CycleTracker:
    """Count a stroke only after a loaded backward push and airborne return."""
    def __init__(self, n, device):
        self.previous = torch.ones(n, 2, dtype=torch.bool, device=device)
        for name in ('back', 'impulse', 'stance_time', 'loaded_time', 'air', 'peak_air', 'forward', 'pending_age', 'cycles', 'unsupported_air'):
            setattr(self, name, torch.zeros(n, 2, device=device))
        self.pushed = torch.zeros_like(self.previous)
        self.pending = torch.zeros_like(self.previous)
        self.bonus = torch.zeros(n, device=device)

    def reset(self, ids):
        self.previous[ids] = True
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and name != 'previous': value[ids] = 0

    def update(self, stance, normal, relative_vx, force_x, body_vx, dt):
        recovery_start = self.previous & ~stance
        touchdown_phase = ~self.previous & stance
        valid_push = (self.back > .06)&(self.impulse > .5)&(self.loaded_time > .6*self.stance_time)
        self.pushed = torch.where(recovery_start, valid_push, self.pushed)
        valid_return = self.pushed&(self.peak_air >= .04)&(self.peak_air <= .40)&(self.forward > .06)&(self.unsupported_air < .04)
        self.pending |= touchdown_phase & valid_return
        self.pending_age = torch.where(touchdown_phase, 0., self.pending_age+dt)
        landed = self.pending&(normal > 60.)&(self.pending_age < .16)&((body_vx > .10)[:, None])
        self.cycles += landed.float(); self.bonus.copy_(landed.float().sum(-1))
        self.pending &= ~landed & (self.pending_age < .16)
        for name in ('back', 'impulse', 'stance_time', 'loaded_time'):
            value = getattr(self, name); value[touchdown_phase] = 0.
        for name in ('air', 'peak_air', 'forward', 'unsupported_air'):
            value = getattr(self, name); value[recovery_start] = 0.
        loaded = normal > 60.
        self.stance_time += stance*dt; self.loaded_time += stance*loaded*dt
        self.back += stance*loaded*(-relative_vx).clamp_min(0)*dt
        self.impulse += stance*loaded*force_x.clamp_min(0)*dt
        airborne = (~stance)&(normal < 5.)
        self.unsupported_air += airborne*(normal.flip(-1) < 60.)*dt
        self.air = torch.where(airborne, self.air+dt, 0.)
        self.peak_air = torch.maximum(self.peak_air, self.air)
        self.forward += airborne*relative_vx.clamp_min(0)*dt
        self.previous.copy_(stance)


def failure_mask(height, upright, illegal, local_xy):
    # A normal 20-second trial at .4 m/s travels 8 m; the base 4-m
    # standing guard would incorrectly terminate every successful skater.
    return (height < .55) | (upright < .5) | illegal | (local_xy.norm(dim=-1) > 12.)
