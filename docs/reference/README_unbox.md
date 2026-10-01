# Unbox / sit-up imitation reference (AgiBot Expedition A3)

Source short: [youtube.com/shorts/qCQSAEAf3Js](https://youtube.com/shorts/qCQSAEAf3Js)
(~14 s, crate open → sit on lip → stand). Q1 ends the unbox PPO at a **four-wheel kneel**;
`Isaac-Q1-Posture-v0` does the real stand.

## Artifacts

| file | what |
|---|---|
| `a3_unbox_ref.mp4` | downloaded short (720×1280) |
| `a3_unbox_ref_pose_overlay.mp4` | MediaPipe skeleton overlay |
| `a3_unbox_ref_pose_angles.csv` | per-frame MediaPipe sagittal angles |
| `a3_unbox_ref_q1_trajectory.json` | per-frame Q1 joint dict (web scrubber) |
| `a3_unbox_ref_q1_keyframes.json` | 5 knots: lie/tuck seeded, sit/rise from MediaPipe (+ yoga arms), kneel seeded |
| `a3_unbox_ref_web_recording.json` | **your** motor angles after editing in the web UI |
| `unbox_knots_from_web.json` | sidecar written by `apply_unbox_recording.py` |

## Workflow (MediaPipe → web motors → RL)

```bash
# 0. deps (once)
python -m venv ~/projects/q1_wheel/.venv_media && . ~/projects/q1_wheel/.venv_media/bin/activate
pip install yt-dlp mediapipe opencv-python-headless numpy
mkdir -p tools/models
curl -L -o tools/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# 1. download + extract
yt-dlp -f "136+140/best" --merge-output-format mp4 \
  -o docs/reference/a3_unbox_ref.mp4 https://youtube.com/shorts/qCQSAEAf3Js
python tools/unbox_pose_extract.py \
  --video docs/reference/a3_unbox_ref.mp4 \
  --model tools/models/pose_landmarker_heavy.task \
  --out docs/reference --overlay

# 2. scrub / edit motor angles on the URDF in the browser
python web/serve.py   # http://127.0.0.1:8765/web/
#    → Load A3 keyframes → adjust sliders → Capture knot → Save for RL

# 3. push recorded angles into Isaac unbox rewards
python tools/apply_unbox_recording.py \
  --recording docs/reference/a3_unbox_ref_web_recording.json

# 4. train
./train_unbox.sh
```

## Why seed lie / kneel?

MediaPipe PoseLandmarker is a **human** model. On this top-down crate shot the early frames
often look “already upright”, so hip/shoulder signs flip. We therefore:

1. **Time-sample** the clip (not sort by `torso_up`).
2. **Anchor** `lie` / `tuck` / `kneel` to measured Q1 poses.
3. Keep MediaPipe for mid-clip `sit` / `rise`, but force arms-back (`shoulder_pitch ≈ 2.2`)
   when the detector reports arms-forward — that matches the yoga-press sit-up the Q1 needs.
4. Let you **correct every motor on the web UI** and save the recording for RL.

See also the skate reference in this folder (`x2_skate_ref_*`) produced by `tools/x2_pose_extract.py`.
