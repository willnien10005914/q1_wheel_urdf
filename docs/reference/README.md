# Imitation reference: AgiBot Lingxi X2 inline skating

* `x2_skate_ref_1m45s-2m00s.mp4` — 15 s cut (1:45–2:00) of
  [AgiBot Lingxi X2 skating](https://www.youtube.com/watch?v=oJZ8tMIYlY4), 640×360, 30 fps.
  This is the style target: both wheels mostly down, legs swizzle in/out, small forward lean, arms
  trailing behind the torso, head up.
* `x2_skate_ref_1m45s-2m00s_pose_angles.csv` — per-frame sagittal angles from MediaPipe
  PoseLandmarker (heavy model) on the clip, produced by `tools/x2_pose_extract.py`.
  Columns: `torso_lean_deg` (+ = shoulders ahead of hips), `{l,r}_hip_pitch_deg` (+ = flexion),
  `{l,r}_knee_deg` (0 straight, + flexed), `{l,r}_shoulder_pitch_deg` (+ = arm behind torso),
  `{l,r}_elbow_deg`.
* `x2_skate_ref_1m45s-2m00s_q1_keyframe.json` — medians mapped to Q1 joint sign conventions.

## What the extraction says (365 / 450 frames tracked)

| quantity                 | median  | IQR           | Q1 target used in training                          |
|--------------------------|--------:|---------------|-----------------------------------------------------|
| knee flexion             | 65°     | 43°–80°       | default stance knee = 0.70 rad (40°), policy free   |
| hip flexion              | 23°     | 10°–37°       | default stance hip pitch = −0.35 rad                |
| torso lean               | −2°     | −9°…+4°       | spec band `projected_gravity_b[0]` 0.12–0.22 (lean reward target 0.17) |
| shoulder pitch (back +)  | −7°     | −25°…+11°     | arms-back keyframe shoulder pitch = +0.55 rad, elbow −0.50 rad |
| elbow flexion            | 13°     | 6°–25°        | see above                                            |

MediaPipe is a human pose model; on the X2 the leg chain tracks well (knee/hip are usable), the
shoulder/elbow landmarks are noisy because the arms are thin yellow paddles behind the body and the
torso lean is small in this clip. For the arm keyframe and the lean band we therefore follow the task
spec / visual inspection of the frames rather than the raw medians. The CSV is kept as the ground
truth trace for a later motion-imitation (AMP / DeepMimic style) reward if the keyframe approach is
not enough.

Regenerate:

```bash
python -m venv .venv_media && . .venv_media/bin/activate
pip install yt-dlp mediapipe opencv-python-headless numpy
yt-dlp --download-sections "*1:45-2:00" -f "b[height<=720][ext=mp4]" -o x2.mp4 https://www.youtube.com/watch?v=oJZ8tMIYlY4
curl -o pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
python tools/x2_pose_extract.py --video docs/reference/x2_skate_ref_1m45s-2m00s.mp4 --model pose_landmarker_heavy.task --out docs/reference --overlay
```
