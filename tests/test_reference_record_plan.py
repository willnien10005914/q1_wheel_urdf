"""The delivered clips must expose each action without hiding its preparation."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reference_record_plan import MotionClip, recording_plan


class RecordingPlanTest(unittest.TestCase):
    def test_all_five_skills_and_both_original_sequences_are_recorded(self):
        plan = recording_plan()
        self.assertEqual(len(plan), 7)
        self.assertEqual({clip.focus_skill for clip in plan} - {'combined'}, {'standup', 'kneel', 'walk', 'backflip', 'skate'})
        self.assertTrue({'standup', 'slow_kneel', 'forward_glide', 'backflip', 'skating',
                         'crouch_rise_glide_backflip', 'alternating_skating'} <= {clip.name for clip in plan})

    def test_every_clip_rises_from_boot_without_reset_or_pose_teleport(self):
        for clip in recording_plan():
            self.assertEqual(clip.events[0][1], 'standup')
            self.assertGreater(clip.events[0][0], 0.)
            self.assertFalse({'reset', 'kneel_hold'} & {mode for _, mode in clip.events})

    def test_slow_kneel_preserves_full_rise_descent_and_hold(self):
        for duration in (8., 24., 39.12):
            clip = next(c for c in recording_plan(duration) if c.name == 'slow_kneel')
            rise, kneel = clip.events
            self.assertGreaterEqual(kneel[0] - rise[0], 13.)
            self.assertGreaterEqual(clip.duration_s - kneel[0], duration + 4. - 1.e-9)
            self.assertEqual(clip.velocity_windows, ())

    def test_exact_motor_action_times_and_stop_window(self):
        clip = next(c for c in recording_plan() if c.name == 'crouch_rise_glide_backflip')
        self.assertEqual(clip.event_steps(.02), {100: 'standup', 750: 'walk', 1100: 'stand_hold', 1250: 'backflip'})
        self.assertEqual(clip.vx(15.), .3)
        self.assertEqual(clip.vx(19.99), .3)
        self.assertEqual(clip.vx(20.), 0.)
        self.assertEqual(clip.vx(24.99), 0.)

    def test_resets_invalid_times_and_shortcut_descent_are_rejected(self):
        for duration in (0., 7.99, math.nan, math.inf):
            with self.assertRaises(ValueError):
                recording_plan(duration)
        for mode in ('reset', 'write_root', 'kneel_hold'):
            with self.assertRaises(ValueError):
                MotionClip('bad', 'kneel', 2., ((1., mode),))
        with self.assertRaises(ValueError):
            recording_plan()[0].event_steps(0.)


if __name__ == '__main__':
    unittest.main()
