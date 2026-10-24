"""CPU-only, deterministic first-trial recording schedules.

Each clip starts in the same physical boot crouch and rises on motor commands.
The recorder may reset only once before a clip, never between its actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MotionClip:
    name: str
    focus_skill: str
    duration_s: float
    events: tuple[tuple[float, str], ...]
    velocity_windows: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self):
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('Clip duration must be finite and positive')
        previous = -1.
        modes = {'standup', 'kneel', 'walk', 'stand_hold', 'backflip', 'slide'}
        for at, mode in self.events:
            if not math.isfinite(at) or not previous < at < self.duration_s or mode not in modes:
                raise ValueError('Clip actions must be ordered, in bounds, motor-only commands')
            previous = at
        for begin, end, vx in self.velocity_windows:
            if not all(math.isfinite(v) for v in (begin, end, vx)) or not 0 <= begin < end <= self.duration_s or not 0 <= vx <= .4:
                raise ValueError('Invalid movement window')

    def event_steps(self, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Simulation step must be finite and positive')
        result = {round(at / dt): mode for at, mode in self.events}
        if len(result) != len(self.events):
            raise ValueError('Simulation step merges recording actions')
        return result

    def vx(self, time_s: float):
        return next((vx for begin, end, vx in self.velocity_windows if begin <= time_s < end), 0.)


def recording_plan(kneel_duration_s=24.):
    """Five individual motion clips plus the two requested combined examples.

    Slow-kneel duration follows the task's actual retimed trajectory. Leave four
    additional seconds to demonstrate a supported final crouch, with all rise
    and descent frames retained in the same uncut trial.
    """
    if not math.isfinite(kneel_duration_s) or kneel_duration_s < 8.:
        raise ValueError('Slow kneel must preserve at least eight seconds of descent')
    return (
        MotionClip('standup', 'standup', 17., ((1., 'standup'),)),
        MotionClip('slow_kneel', 'kneel', 18. + kneel_duration_s, ((1., 'standup'), (14., 'kneel'))),
        MotionClip('forward_glide', 'walk', 28., ((1., 'standup'), (14., 'walk')), ((14., 25., .3),)),
        MotionClip('backflip', 'backflip', 28., ((1., 'standup'), (14., 'stand_hold'), (17., 'backflip'))),
        MotionClip('skating', 'skate', 28., ((1., 'standup'), (14., 'slide')), ((14., 25., .3),)),
        MotionClip('crouch_rise_glide_backflip', 'combined', 36.,
                   ((2., 'standup'), (15., 'walk'), (22., 'stand_hold'), (25., 'backflip')), ((15., 20., .3),)),
        MotionClip('alternating_skating', 'skate', 28., ((1., 'standup'), (14., 'slide')), ((14., 25., .3),)),
    )
