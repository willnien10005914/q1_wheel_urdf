#!/usr/bin/env python3
"""Extract an unbox / sit-up joint-angle reference for Q1 from an AgiBot A3 clip.

Pipeline
  1. MediaPipe PoseLandmarker (tasks API, world landmarks) on every frame.
  2. Full-body angles from 3-D landmarks (works for top-down / 3/4 unbox shots):
       torso_lean_deg   : hip→shoulder vs vertical, + = shoulders ahead of hips
       {l,r}_hip_pitch  : thigh flexion (+ flexed)
       {l,r}_knee       : knee flexion (0 straight, + flexed)
       {l,r}_shoulder_pitch : upper-arm vs torso, + = arm behind torso (Q1 +Y)
       {l,r}_elbow      : elbow flexion (0 straight, + flexed)
  3. Map to Q1 URDF joint signs and emit:
       *_pose_angles.csv          per-frame MediaPipe angles
       *_q1_trajectory.json       per-frame Q1 joint dict + phase
       *_q1_keyframes.json        sparse knots for web UI + Isaac `getup_track`
       *_pose_overlay.mp4         optional skeleton overlay

The A3 has feet, Q1 has wheels: the final stand keyframe is remapped to the Q1 four-wheel
kneel (hip/knee/waist from ``wheel_humanoid`` kneel constants) so unbox PPO still ends at
kneel and posture PPO does the stand.

Usage:
  python tools/unbox_pose_extract.py \\
      --video docs/reference/a3_unbox_ref.mp4 \\
      --model tools/models/pose_landmarker_heavy.task \\
      --out docs/reference --overlay
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

# MediaPipe pose landmark ids
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28

# Q1 kneel targets (must match wheel_humanoid_lab.assets.wheel_humanoid)
KNEEL = {
    "l_hip_pitch_joint": -1.02,
    "r_hip_pitch_joint": -1.02,
    "l_knee_joint": 2.36,
    "r_knee_joint": 2.36,
    "l_knee_roller_joint": 0.444444 * 2.36,
    "r_knee_roller_joint": 0.444444 * 2.36,
    "waist_pitch_joint": 0.30,
    "l_shoulder_pitch_joint": 0.55,
    "r_shoulder_pitch_joint": 0.55,
    "l_shoulder_roll_joint": 0.20,
    "r_shoulder_roll_joint": -0.20,
    "l_elbow_joint": -0.50,
    "r_elbow_joint": -0.50,
}

# Soft joint limits used when clamping MediaPipe → URDF
LIMITS = {
    "waist_pitch_joint": (-0.52, 0.52),
    "l_hip_pitch_joint": (-1.8, 1.2),
    "r_hip_pitch_joint": (-1.8, 1.2),
    "l_knee_joint": (-0.2, 2.6),
    "r_knee_joint": (-0.2, 2.6),
    "l_shoulder_pitch_joint": (-3.14, 3.14),
    "r_shoulder_pitch_joint": (-3.14, 3.14),
    "l_elbow_joint": (-2.5, 0.2),
    "r_elbow_joint": (-2.5, 0.2),
}


def _ang(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))))


def body_angles(P: np.ndarray) -> dict | None:
    """P: (33, 3) world landmarks, MediaPipe: x right, y DOWN, z toward camera."""
    up = np.array([0.0, -1.0, 0.0])
    hip = 0.5 * (P[L_HIP] + P[R_HIP])
    sh = 0.5 * (P[L_SH] + P[R_SH])
    torso = sh - hip
    if np.linalg.norm(torso) < 0.05:
        return None

    # Forward ≈ projection of hip→ankle mean onto horizontal, else shoulder-line normal.
    ankle = 0.5 * (P[L_ANK] + P[R_ANK])
    fwd = ankle - hip
    fwd[1] = 0.0
    if np.linalg.norm(fwd) < 1e-3:
        shoulder_line = P[L_SH] - P[R_SH]
        fwd = np.cross(shoulder_line, up)
        fwd[1] = 0.0
    if np.linalg.norm(fwd) < 1e-3:
        return None
    fwd /= np.linalg.norm(fwd)

    def sag(v):
        return np.array([np.dot(v, fwd), np.dot(v, up)])

    def signed_from_up(v2):
        return math.degrees(math.atan2(v2[0], v2[1]))

    def fwd_from_down(v2):
        return math.degrees(math.atan2(v2[0], -v2[1]))

    torso_lean = signed_from_up(sag(torso))
    # Proxy for how upright the trunk is: 1 = standing (torso along -y), 0 = supine.
    torso_up = float(np.clip(np.dot(torso / (np.linalg.norm(torso) + 1e-9), up), 0.0, 1.0))
    # Hip height proxy in MediaPipe world (y down): smaller y = higher off the floor.
    hip_y = float(hip[1])

    out = {
        "torso_lean_deg": torso_lean,
        "torso_up": torso_up,
        "hip_y": hip_y,
    }
    for side, (H, K, A, S, E, W) in {
        "l": (L_HIP, L_KNEE, L_ANK, L_SH, L_EL, L_WR),
        "r": (R_HIP, R_KNEE, R_ANK, R_SH, R_EL, R_WR),
    }.items():
        thigh = sag(P[K] - P[H])
        shank = sag(P[A] - P[K])
        uarm = sag(P[E] - P[S])
        farm = sag(P[W] - P[E])
        out[f"{side}_hip_pitch_deg"] = fwd_from_down(thigh) + torso_lean
        out[f"{side}_knee_deg"] = 180.0 - _ang(-thigh, shank)
        out[f"{side}_shoulder_pitch_deg"] = -fwd_from_down(uarm) + torso_lean
        out[f"{side}_elbow_deg"] = 180.0 - _ang(-uarm, farm)
    return out


def clamp_joint(name: str, value: float) -> float:
    lo, hi = LIMITS.get(name, (-math.pi, math.pi))
    return float(np.clip(value, lo, hi))


def angles_to_q1(a: dict) -> dict:
    """Map MediaPipe degrees → Q1 URDF radians (sign conventions from x2_pose_extract)."""
    d2r = math.pi / 180.0
    hip = 0.5 * (a["l_hip_pitch_deg"] + a["r_hip_pitch_deg"])
    knee = 0.5 * (a["l_knee_deg"] + a["r_knee_deg"])
    sh = 0.5 * (a["l_shoulder_pitch_deg"] + a["r_shoulder_pitch_deg"])
    el = 0.5 * (a["l_elbow_deg"] + a["r_elbow_deg"])
    # Waist pitch proxy: when torso is upright, lean maps to waist; when supine, keep near 0.
    waist = a["torso_lean_deg"] * d2r * a["torso_up"]
    # Q1: hip +y negative = thigh forward; knee +y = flexion; shoulder +y = arm back; elbow - = flex.
    q = {
        "waist_pitch_joint": clamp_joint("waist_pitch_joint", waist),
        "l_hip_pitch_joint": clamp_joint("l_hip_pitch_joint", -hip * d2r),
        "r_hip_pitch_joint": clamp_joint("r_hip_pitch_joint", -hip * d2r),
        "l_knee_joint": clamp_joint("l_knee_joint", knee * d2r),
        "r_knee_joint": clamp_joint("r_knee_joint", knee * d2r),
        "l_knee_roller_joint": clamp_joint("l_knee_joint", knee * d2r) * 0.444444,
        "r_knee_roller_joint": clamp_joint("r_knee_joint", knee * d2r) * 0.444444,
        "l_shoulder_pitch_joint": clamp_joint("l_shoulder_pitch_joint", sh * d2r),
        "r_shoulder_pitch_joint": clamp_joint("r_shoulder_pitch_joint", sh * d2r),
        "l_shoulder_roll_joint": 0.20,
        "r_shoulder_roll_joint": -0.20,
        "l_elbow_joint": clamp_joint("l_elbow_joint", -el * d2r),
        "r_elbow_joint": clamp_joint("r_elbow_joint", -el * d2r),
        "projected_gravity_x": math.sin(a["torso_lean_deg"] * d2r),
        "torso_up": a["torso_up"],
    }
    return q


def phase_from_torso(torso_up: float) -> float:
    """Map MediaPipe uprightness to unbox phase 0=supine … 1=kneel/stand."""
    # Empirically: supine ~0.05–0.2, sit ~0.4–0.7, stand ~0.85–1.0
    return float(np.clip((torso_up - 0.05) / 0.90, 0.0, 1.0))


# Authoritative Q1 box-open / kneel seeds. MediaPipe on top-down crate shots is noisy for the
# supine start; we anchor endpoints and let mid-clip MediaPipe + web editing fill the middle.
LIE_SEED = {
    "waist_pitch_joint": 0.0,
    "l_hip_pitch_joint": -0.4,
    "r_hip_pitch_joint": -0.4,
    "l_knee_joint": 0.4,
    "r_knee_joint": 0.4,
    "l_knee_roller_joint": 0.444444 * 0.4,
    "r_knee_roller_joint": 0.444444 * 0.4,
    "l_shoulder_pitch_joint": 2.2,
    "r_shoulder_pitch_joint": 2.2,
    "l_shoulder_roll_joint": 0.3,
    "r_shoulder_roll_joint": -0.3,
    "l_elbow_joint": -1.6,
    "r_elbow_joint": -1.6,
}
TUCK_SEED = {
    **LIE_SEED,
    "l_hip_pitch_joint": -1.15,
    "r_hip_pitch_joint": -1.15,
    "l_knee_joint": 2.20,
    "r_knee_joint": 2.20,
    "l_knee_roller_joint": 0.444444 * 2.20,
    "r_knee_roller_joint": 0.444444 * 2.20,
}


def pick_keyframes(frames: list[dict], n: int = 5) -> list[dict]:
    """Evenly sample along *time* (crate unbox is chronological), not torso_up rank.

    Top-down MediaPipe often reports the boxed robot as already upright, so sorting by
    ``phase`` scrambled lie/tuck. Time order matches the A3 short.
    """
    if not frames:
        return []
    ordered = sorted(frames, key=lambda f: f["t_s"])
    if len(ordered) <= n:
        return ordered
    idxs = [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
    return [ordered[i] for i in idxs]


def pelvis_z_for_phase(phase: float) -> float:
    # Lie ~0.22, tuck/yoga ~0.30–0.38, kneel 0.45
    return float(0.22 + 0.23 * phase)


def knot_pose(q: dict, *, force_kneel: bool = False) -> dict:
    """Regex pose map for Isaac getup_track / web presets."""
    if force_kneel:
        return {
            ".*_hip_pitch_joint": KNEEL["l_hip_pitch_joint"],
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": KNEEL["l_knee_joint"],
            ".*_knee_roller_joint": KNEEL["l_knee_roller_joint"],
            "waist_pitch_joint": KNEEL["waist_pitch_joint"],
            ".*_shoulder_pitch_joint": KNEEL["l_shoulder_pitch_joint"],
            "l_shoulder_roll_joint": KNEEL["l_shoulder_roll_joint"],
            "r_shoulder_roll_joint": KNEEL["r_shoulder_roll_joint"],
            ".*_elbow_joint": KNEEL["l_elbow_joint"],
        }
    return {
        ".*_hip_pitch_joint": q["l_hip_pitch_joint"],
        ".*_hip_roll_joint": 0.0,
        ".*_knee_joint": q["l_knee_joint"],
        ".*_knee_roller_joint": q["l_knee_roller_joint"],
        "waist_pitch_joint": q["waist_pitch_joint"],
        ".*_shoulder_pitch_joint": q["l_shoulder_pitch_joint"],
        "l_shoulder_roll_joint": q["l_shoulder_roll_joint"],
        "r_shoulder_roll_joint": q["r_shoulder_roll_joint"],
        ".*_elbow_joint": q["l_elbow_joint"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="pose_landmarker_*.task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_conf", type=float, default=0.25)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--n_keyframes", type=int, default=5)
    ap.add_argument(
        "--source_url",
        default="https://youtube.com/shorts/qCQSAEAf3Js",
        help="Attribution URL stored in the JSON metadata",
    )
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
        writer = cv2.VideoWriter(
            os.path.join(args.out, f"{base}_pose_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    rows = []
    traj = []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(idx * 1000 / fps)
            )
            if res.pose_world_landmarks:
                P = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                vis = np.array([p.visibility for p in res.pose_landmarks[0]])
                a = body_angles(P)
                if a is not None and vis[[L_HIP, R_HIP, L_KNEE, R_KNEE, L_SH, R_SH]].min() > 0.15:
                    a["frame"] = idx
                    a["t_s"] = idx / fps
                    rows.append(a)
                    q = angles_to_q1(a)
                    phase = phase_from_torso(a["torso_up"])
                    traj.append(
                        {
                            "frame": idx,
                            "t_s": round(idx / fps, 4),
                            "phase": round(phase, 4),
                            "joints": {k: round(v, 4) for k, v in q.items() if k.endswith("_joint")},
                            "meta": {
                                "torso_up": round(a["torso_up"], 4),
                                "torso_lean_deg": round(a["torso_lean_deg"], 3),
                                "projected_gravity_x": round(q["projected_gravity_x"], 4),
                            },
                        }
                    )
                if writer is not None and res.pose_landmarks:
                    for p in res.pose_landmarks[0]:
                        cv2.circle(frame, (int(p.x * W), int(p.y * H)), 3, (0, 255, 255), -1)
                    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS:
                        p0, p1 = res.pose_landmarks[0][c.start], res.pose_landmarks[0][c.end]
                        cv2.line(
                            frame,
                            (int(p0.x * W), int(p0.y * H)),
                            (int(p1.x * W), int(p1.y * H)),
                            (0, 200, 0),
                            2,
                        )
            if writer is not None:
                writer.write(frame)
            idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    if not rows:
        raise SystemExit("no poses detected — try lowering --min_conf")

    keys = ["frame", "t_s", "torso_lean_deg", "torso_up", "hip_y"] + [
        k for k in rows[0] if k not in ("frame", "t_s", "torso_lean_deg", "torso_up", "hip_y")
    ]
    csv_path = os.path.join(args.out, f"{base}_pose_angles.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in keys})

    traj_path = os.path.join(args.out, f"{base}_q1_trajectory.json")
    traj_doc = {
        "source": os.path.basename(args.video),
        "source_url": args.source_url,
        "fps": fps,
        "frames_used": len(traj),
        "frames_total": idx,
        "note": (
            "MediaPipe→Q1 joint trajectory for web joint playback. Final stand is remapped to "
            "Q1 four-wheel kneel in the keyframe export (A3 has feet, Q1 has wheels)."
        ),
        "trajectory": traj,
    }
    with open(traj_path, "w") as f:
        json.dump(traj_doc, f, indent=2)

    picks = pick_keyframes(traj, n=args.n_keyframes)
    labels = ["lie", "tuck", "sit", "rise", "kneel"]
    knots = []
    for i, fr in enumerate(picks):
        label = labels[min(i, len(labels) - 1)] if len(picks) >= 5 else f"k{i}"
        force_kneel = label == "kneel" or i == len(picks) - 1
        # Anchor endpoints: MediaPipe is unreliable while the robot is still in the crate.
        if label == "lie":
            joints = dict(LIE_SEED)
            source = "seed_lie"
        elif label == "tuck":
            joints = dict(TUCK_SEED)
            source = "seed_tuck"
        elif force_kneel:
            joints = dict(KNEEL)
            source = "seed_kneel"
        else:
            joints = dict(fr["joints"])
            source = "mediapipe"
            # Yoga press needs arms planted behind the torso (+ shoulder pitch). MediaPipe often
            # reports arms-forward on this short; keep the unbox yoga shoulder/elbow when flipped.
            if joints.get("l_shoulder_pitch_joint", 0.0) < 0.8:
                joints["l_shoulder_pitch_joint"] = 2.2
                joints["r_shoulder_pitch_joint"] = 2.2
                joints["l_elbow_joint"] = -1.4
                joints["r_elbow_joint"] = -1.4
                source = "mediapipe+yoga_arms"
            # Waist must pitch the trunk up for the sit-up; top-down lean signs flip easily.
            if joints.get("waist_pitch_joint", 0.0) < 0.15:
                joints["waist_pitch_joint"] = 0.50 if label == "sit" else 0.35
                source = source + "+waist"
        pose = knot_pose(joints, force_kneel=force_kneel)
        phase = i / max(len(picks) - 1, 1)
        knots.append(
            {
                "phase": round(phase, 4),
                "t_s": fr["t_s"],
                "frame": fr["frame"],
                "pelvis_z": round(pelvis_z_for_phase(1.0 if force_kneel else phase), 3),
                "joints": {k: round(v, 4) for k, v in joints.items()},
                "joints_mediapipe": {k: round(v, 4) for k, v in fr["joints"].items()},
                "pose_regex": pose,
                "label": label,
                "source": source,
            }
        )

    # Isaac-compatible knot list: (phase, pose_regex, pelvis_z)
    isaac_knots = [[k["phase"], k["pose_regex"], k["pelvis_z"]] for k in knots]

    kf_path = os.path.join(args.out, f"{base}_q1_keyframes.json")
    kf_doc = {
        "source": os.path.basename(args.video),
        "source_url": args.source_url,
        "frames_used": len(traj),
        "keyframes": knots,
        "isaac_knots": isaac_knots,
        "web_preset": {k["label"]: k["joints"] for k in knots},
        "workflow": [
            "1. python tools/unbox_pose_extract.py --video ... --model ... --out docs/reference --overlay",
            "2. python web/serve.py  → open /web/  → Load keyframes JSON → scrub / edit joints",
            "3. Record motor angles (Download recording) → docs/reference/*_web_recording.json",
            "4. python tools/apply_unbox_recording.py --recording ...  (updates UNBOX_KNOTS)",
            "5. ./train_unbox.sh",
        ],
    }
    with open(kf_path, "w") as f:
        json.dump(kf_doc, f, indent=2)

    print(
        json.dumps(
            {
                "csv": csv_path,
                "trajectory": traj_path,
                "keyframes": kf_path,
                "frames_used": len(traj),
                "keyframes_n": len(knots),
                "phase_range": [traj[0]["phase"], traj[-1]["phase"]],
                "labels": [k["label"] for k in knots],
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    print(kf_path)


if __name__ == "__main__":
    main()
