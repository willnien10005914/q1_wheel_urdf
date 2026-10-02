#!/bin/bash
# Record supine + prone + slide (L/R fore-aft) in Isaac Sim, upload to gofile, print links.
#   ./tools/record_and_upload_demo_videos.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"
export PYTHONPATH="$ROOT/source/wheel_humanoid_lab:${PYTHONPATH:-}"
export WHEEL_HUMANOID_ROOT="$ROOT"
python -m pip install -e source/wheel_humanoid_lab -q

STEPS_REC="${STEPS_REC:-2999}"
STEPS_SLIDE="${STEPS_SLIDE:-800}"
SUPINE_CKPT="${SUPINE_CKPT:-$ROOT/checkpoints/q1_recovery_v3_supine_ppo.pt}"
PRONE_CKPT="${PRONE_CKPT:-$ROOT/checkpoints/q1_recovery_v3_prone_ppo.pt}"
SLIDE_CKPT="${SLIDE_CKPT:-$ROOT/checkpoints/q1_slide_ppo.pt}"
SUPINE_SEED="${SUPINE_SEED:-5107}"
PRONE_SEED="${PRONE_SEED:-4119}"

mkdir -p docs/reference/review_recovery_v3/split_supine \
         docs/reference/review_recovery_v3/split_prone \
         docs/results \
         docs/demos

echo "=== [1/3] SUPINE recovery video (seed=$SUPINE_SEED, steps=$STEPS_REC) ==="
bash evaluate_recovery_v3.sh \
  --checkpoint "$SUPINE_CKPT" --mode supine --seed "$SUPINE_SEED" \
  --video --steps "$STEPS_REC" \
  --out docs/reference/review_recovery_v3/split_supine

echo "=== [2/3] PRONE recovery video (seed=$PRONE_SEED, steps=$STEPS_REC) ==="
bash evaluate_recovery_v3.sh \
  --checkpoint "$PRONE_CKPT" --mode prone --seed "$PRONE_SEED" \
  --video --steps "$STEPS_REC" \
  --out docs/reference/review_recovery_v3/split_prone

echo "=== [3/3] SLIDE L/R fore-aft video (steps=$STEPS_SLIDE) ==="
OUT="$ROOT/docs/results/q1_slide_foreaft_live.mp4" \
  CHECKPOINT="$SLIDE_CKPT" STEPS="$STEPS_SLIDE" \
  bash record_slide.sh

echo "=== Upload to gofile ==="
python3 tools/upload_gofile.py \
  docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.mp4 \
  docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.gofile.json
cp -f docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.gofile.json \
  docs/reference/review_recovery_v3/split_supine.gofile.json

python3 tools/upload_gofile.py \
  docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.mp4 \
  docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.gofile.json
cp -f docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.gofile.json \
  docs/reference/review_recovery_v3/split_prone.gofile.json

python3 tools/upload_gofile.py \
  docs/results/q1_slide_foreaft_live.mp4 \
  docs/results/q1_slide_foreaft_live.gofile.json

python3 - <<'PY'
import json
from pathlib import Path
rows = [
    ("正躺 supine", "checkpoints/q1_recovery_v3_supine_ppo.pt",
     "docs/reference/review_recovery_v3/split_supine/recovery_eval_supine.gofile.json"),
    ("趴躺 prone", "checkpoints/q1_recovery_v3_prone_ppo.pt",
     "docs/reference/review_recovery_v3/split_prone/recovery_eval_prone.gofile.json"),
    ("側滑 左右腳前後 slide", "checkpoints/q1_slide_ppo.pt",
     "docs/results/q1_slide_foreaft_live.gofile.json"),
]
lines = ["# Live Isaac Sim demo uploads\n", "| 動作 | PPO | gofile |\n|---|---|---|\n"]
print("\n======== gofile links ========")
for name, ckpt, gj in rows:
    url = json.loads(Path(gj).read_text())["data"]["downloadPage"]
    lines.append(f"| {name} | `{ckpt}` | {url} |\n")
    print(f"{name}\n  PPO: {ckpt}\n  {url}\n")
Path("docs/demos/LIVE_UPLOADS.md").write_text("".join(lines))
print("wrote docs/demos/LIVE_UPLOADS.md")
PY
echo "=== DONE $(date -Is) ==="
