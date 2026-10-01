#!/usr/bin/env python3
"""Apply a web-recorded unbox motor-angle JSON into ``unbox_env_cfg`` pose constants.

Reads ``docs/reference/*_web_recording.json`` (from the joint UI "Save for RL" button) and
rewrites ``LIE_POSE`` / ``TUCK_POSE`` / ``YOGA_POSE`` / ``KNEEL_POSE`` / ``UNBOX_KNOTS``.
Also writes ``docs/reference/unbox_knots_from_web.json`` as a sidecar.

Usage:
  python tools/apply_unbox_recording.py \\
      --recording docs/reference/a3_unbox_ref_web_recording.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/unbox/unbox_env_cfg.py"


def _pose_regex(joints: dict) -> dict:
    def avg(a, b):
        return 0.5 * (float(joints.get(a, 0.0)) + float(joints.get(b, 0.0)))

    hip = avg("l_hip_pitch_joint", "r_hip_pitch_joint")
    knee = avg("l_knee_joint", "r_knee_joint")
    sh = avg("l_shoulder_pitch_joint", "r_shoulder_pitch_joint")
    el = avg("l_elbow_joint", "r_elbow_joint")
    return {
        ".*_hip_pitch_joint": round(hip, 4),
        ".*_hip_roll_joint": 0.0,
        ".*_knee_joint": round(knee, 4),
        ".*_knee_roller_joint": round(knee * 0.444444, 4),
        "waist_pitch_joint": round(float(joints.get("waist_pitch_joint", 0.0)), 4),
        ".*_shoulder_pitch_joint": round(sh, 4),
        "l_shoulder_roll_joint": round(float(joints.get("l_shoulder_roll_joint", 0.2)), 4),
        "r_shoulder_roll_joint": round(float(joints.get("r_shoulder_roll_joint", -0.2)), 4),
        ".*_elbow_joint": round(el, 4),
    }


def _fmt_pose(name: str, pose: dict) -> str:
    lines = [f"{name} = {{"]
    for k, v in pose.items():
        lines.append(f'    "{k}": {v},')
    lines.append("}")
    return "\n".join(lines)


def _replace_assign(src: str, name: str, body: str) -> str:
    pat = re.compile(rf"{name} = \{{.*?\n\}}", re.S)
    if not pat.search(src):
        raise SystemExit(f"could not find {name} in env cfg")
    return pat.sub(body, src, count=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--env-cfg", type=Path, default=ENV_CFG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = json.loads(Path(args.recording).read_text())
    kfs = doc.get("keyframes") or []
    if not kfs:
        raise SystemExit("recording has no keyframes")

    norm = []
    for i, k in enumerate(kfs):
        joints = k.get("joints") or {}
        pose = k.get("pose_regex") or _pose_regex(joints)
        norm.append(
            {
                "phase": float(k.get("phase", i / max(len(kfs) - 1, 1))),
                "pose_regex": pose,
                "pelvis_z": float(k.get("pelvis_z", 0.22 + 0.23 * float(k.get("phase", 0)))),
                "label": k.get("label", f"k{i}"),
                "joints": joints,
            }
        )

    by_label = {n["label"]: n for n in norm}
    lie = by_label.get("lie", norm[0])
    tuck = by_label.get("tuck", norm[min(1, len(norm) - 1)])
    yoga = by_label.get("sit", by_label.get("yoga", by_label.get("rise", norm[min(2, len(norm) - 1)])))
    kneel = by_label.get("kneel", norm[-1])

    lie_p = lie.get("pose_regex") or _pose_regex(lie.get("joints", {}))
    tuck_p = tuck.get("pose_regex") or _pose_regex(tuck.get("joints", {}))
    yoga_p = yoga.get("pose_regex") or _pose_regex(yoga.get("joints", {}))
    kneel_p = kneel.get("pose_regex") or _pose_regex(kneel.get("joints", {}))

    knot_lines = ["UNBOX_KNOTS = ["]
    for i, n in enumerate(norm):
        pose_name = {
            "lie": "LIE_POSE",
            "tuck": "TUCK_POSE",
            "sit": "YOGA_POSE",
            "yoga": "YOGA_POSE",
            "rise": "YOGA_POSE",
            "kneel": "KNEEL_POSE",
        }.get(n["label"], "KNEEL_POSE" if i == len(norm) - 1 else "TUCK_POSE")
        z = "KNEEL_PELVIS_Z" if pose_name == "KNEEL_POSE" else f"{n['pelvis_z']:.2f}"
        phase = 0.0 if i == 0 else (1.0 if i == len(norm) - 1 else n["phase"])
        knot_lines.append(f"    ({phase:.2f}, {pose_name}, {z}),")
    knot_lines.append("]")

    sidecar = {
        "source_recording": Path(args.recording).name,
        "source_url": doc.get("source_url"),
        "LIE_POSE": lie_p,
        "TUCK_POSE": tuck_p,
        "YOGA_POSE": yoga_p,
        "KNEEL_POSE": kneel_p,
        "UNBOX_KNOTS": [[p, pose, z] for p, pose, z in [
            (0.0 if i == 0 else (1.0 if i == len(norm) - 1 else n["phase"]), n["pose_regex"], n["pelvis_z"])
            for i, n in enumerate(norm)
        ]],
    }
    out_json = ROOT / "docs/reference/unbox_knots_from_web.json"

    if args.dry_run:
        print("\n".join(knot_lines))
        print(_fmt_pose("LIE_POSE", lie_p))
        return

    out_json.write_text(json.dumps(sidecar, indent=2) + "\n")
    text = args.env_cfg.read_text()
    text = _replace_assign(text, "LIE_POSE", _fmt_pose("LIE_POSE", lie_p))
    text = _replace_assign(text, "TUCK_POSE", _fmt_pose("TUCK_POSE", tuck_p))
    text = _replace_assign(text, "YOGA_POSE", _fmt_pose("YOGA_POSE", yoga_p))
    text = _replace_assign(text, "KNEEL_POSE", _fmt_pose("KNEEL_POSE", kneel_p))
    knots_pat = re.compile(r"UNBOX_KNOTS = \[.*?\]", re.S)
    if not knots_pat.search(text):
        raise SystemExit("could not find UNBOX_KNOTS")
    text = knots_pat.sub("\n".join(knot_lines), text, count=1)
    args.env_cfg.write_text(text)
    print(f"wrote {out_json}")
    print(f"updated {args.env_cfg}")
    print("labels:", [n["label"] for n in norm])


if __name__ == "__main__":
    main()
