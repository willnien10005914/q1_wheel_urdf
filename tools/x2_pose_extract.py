#!/usr/bin/env python3
"""Extract a skate-style joint-angle reference for Q1 from the AgiBot Lingxi X2 clip.

Pipeline
  1. MediaPipe PoseLandmarker (tasks API, world landmarks in metres) on every frame.
  2. Sagittal-plane angles from the 3-D landmarks (robot is filmed side-on / 3/4):
       torso_lean    : angle of hip->shoulder vector from vertical, + = leaning forward
       hip_pitch     : angle between torso vector and thigh vector (0 = straight, + = thigh flexed forward)
       knee          : 0 = straight, + = flexed
       shoulder_pitch: upper-arm vs torso in the sagittal plane, + = arm behind the torso (arms-back)
       elbow         : 0 = straight, + = flexed
  3. Map to Q1 joint sign conventions (URDF axes) -> CSV per frame + JSON keyframe (median over
     frames where both wheels are on the ground and the robot is upright), used by the Isaac Lab
     skate task as the `arms_back` / `forward_lean` targets.

The X2 is a robot, not a person, so the detector is run at low confidence and frames without a
plausible skeleton are dropped. The result is a *style prior* for reward shaping, not motion
capture ground truth.

Usage:
  python tools/x2_pose_extract.py --video docs/reference/x2_skate_ref.mp4 \
      --model /tmp/pose_landmarker_heavy.task --out docs/reference
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

# MediaPipe pose landmark ids
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28


def _ang(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))))


def sagittal_angles(P: np.ndarray) -> dict | None:
    """P: (33, 3) world landmarks, MediaPipe frame: x right, y DOWN, z toward camera (negative = away)."""
    up = np.array([0.0, -1.0, 0.0])
    hip = 0.5 * (P[L_HIP] + P[R_HIP])
    sh = 0.5 * (P[L_SH] + P[R_SH])
    torso = sh - hip
    if np.linalg.norm(torso) < 0.15:
        return None
    # Robot forward = horizontal component of the hip->knee mean direction... unreliable while
    # skating; use the shoulder line normal instead (side view => normal points along travel).
    shoulder_line = P[L_SH] - P[R_SH]
    fwd = np.cross(shoulder_line, up)
    fwd[1] = 0.0
    if np.linalg.norm(fwd) < 1e-3:
        return None
    fwd /= np.linalg.norm(fwd)
    # Disambiguate forward: in the X2 skate stance the hips are flexed, so the knees sit ahead of
    # the hips (much more robust than the shin direction, which swings during a swizzle).
    knee_ahead = 0.5 * ((P[L_KNEE] - P[L_HIP]) + (P[R_KNEE] - P[R_HIP]))
    if np.dot(knee_ahead, fwd) < 0:
        fwd = -fwd

    def sag(v):
        # project onto sagittal plane (fwd, up)
        return np.array([np.dot(v, fwd), np.dot(v, up)])

    def signed_from_up(v2):
        # angle from +up toward +fwd, degrees
        return math.degrees(math.atan2(v2[0], v2[1]))

    t2 = sag(torso)
    torso_lean = signed_from_up(t2)  # + when shoulders ahead of hips

    out = {"torso_lean_deg": torso_lean}
    for side, (H, K, A, S, E, W) in {
        "l": (L_HIP, L_KNEE, L_ANK, L_SH, L_EL, L_WR),
        "r": (R_HIP, R_KNEE, R_ANK, R_SH, R_EL, R_WR),
    }.items():
        thigh = sag(P[K] - P[H])  # points down(-ish)
        shank = sag(P[A] - P[K])
        uarm = sag(P[E] - P[S])
        farm = sag(P[W] - P[E])

        def fwd_from_down(v2):
            # angle of a downward-pointing segment measured from straight-down toward +fwd
            return math.degrees(math.atan2(v2[0], -v2[1]))

        # hip flexion = thigh ahead of vertical + torso leaning forward (both close the hip angle)
        out[f"{side}_hip_pitch_deg"] = fwd_from_down(thigh) + torso_lean
        out[f"{side}_knee_deg"] = 180.0 - _ang(-thigh, shank)  # 0 straight, + flexed
        # arms-back positive: upper arm behind vertical, plus torso lean (arm trails when leaning)
        out[f"{side}_shoulder_pitch_deg"] = -fwd_from_down(uarm) + torso_lean
        out[f"{side}_elbow_deg"] = 180.0 - _ang(-uarm, farm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="pose_landmarker_*.task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_conf", type=float, default=0.3)
    ap.add_argument("--overlay", action="store_true", help="write an overlay mp4 next to the csv")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_conf,
        min_pose_presence_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf,
        output_segmentation_masks=False,
    )
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    writer = None
    if args.overlay:
        writer = cv2.VideoWriter(os.path.join(args.out, f"{base}_pose_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows = []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(idx * 1000 / fps))
            if res.pose_world_landmarks:
                P = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                vis = np.array([p.visibility for p in res.pose_landmarks[0]])
                a = sagittal_angles(P)
                if a is not None and vis[[L_HIP, R_HIP, L_KNEE, R_KNEE, L_SH, R_SH]].min() > 0.2:
                    a["frame"] = idx
                    a["t_s"] = idx / fps
                    rows.append(a)
                if writer is not None:
                    for p in res.pose_landmarks[0]:
                        cv2.circle(frame, (int(p.x * W), int(p.y * H)), 4, (0, 255, 255), -1)
                    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS:
                        p0, p1 = res.pose_landmarks[0][c.start], res.pose_landmarks[0][c.end]
                        cv2.line(frame, (int(p0.x * W), int(p0.y * H)), (int(p1.x * W), int(p1.y * H)), (0, 200, 0), 2)
            if writer is not None:
                writer.write(frame)
            idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    if not rows:
        raise SystemExit("no poses detected")
    keys = ["frame", "t_s", "torso_lean_deg"] + [k for k in rows[0] if k not in ("frame", "t_s", "torso_lean_deg")]
    with open(os.path.join(args.out, f"{base}_pose_angles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k]) for k in keys})

    def med(k):
        return float(np.median([r[k] for r in rows]))

    def p(k, q):
        return float(np.percentile([r[k] for r in rows], q))

    key = {
        "source": os.path.basename(args.video),
        "frames_used": len(rows),
        "torso_lean_deg": {"median": med("torso_lean_deg"), "p25": p("torso_lean_deg", 25), "p75": p("torso_lean_deg", 75)},
        "hip_pitch_deg": {"median": 0.5 * (med("l_hip_pitch_deg") + med("r_hip_pitch_deg"))},
        "knee_deg": {"median": 0.5 * (med("l_knee_deg") + med("r_knee_deg"))},
        "shoulder_pitch_back_deg": {"median": 0.5 * (med("l_shoulder_pitch_deg") + med("r_shoulder_pitch_deg"))},
        "elbow_deg": {"median": 0.5 * (med("l_elbow_deg") + med("r_elbow_deg"))},
    }
    # ---- Q1 mapping (URDF sign conventions) ------------------------------------------------
    # projected_gravity_b[0] = sin(lean)  (pelvis/torso frame x forward, lean forward -> +)
    # hip pitch axis +y: negative = thigh forward (flexion)  -> q_hip = -hip_pitch
    # knee axis +y: positive = flexion                         -> q_knee = +knee
    # shoulder pitch axis +y: positive = arm backwards           -> q_sh = +shoulder_pitch_back
    # elbow axis +y, limits [-2.5, 0.2]: negative = flexion     -> q_el = -elbow
    d2r = math.pi / 180.0
    key["q1_targets_rad"] = {
        "projected_gravity_x": math.sin(key["torso_lean_deg"]["median"] * d2r),
        "hip_pitch": -key["hip_pitch_deg"]["median"] * d2r,
        "knee": key["knee_deg"]["median"] * d2r,
        "shoulder_pitch": key["shoulder_pitch_back_deg"]["median"] * d2r,
        "elbow": -key["elbow_deg"]["median"] * d2r,
    }
    with open(os.path.join(args.out, f"{base}_q1_keyframe.json"), "w") as f:
        json.dump(key, f, indent=2)
    print(json.dumps(key, indent=2))


if __name__ == "__main__":
    main()
