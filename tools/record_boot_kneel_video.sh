#!/bin/bash
# After boot-kneel full train: record 1-env stand-success video and upload to gofile.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"

CKPT="${BOOT_KNEEL_CKPT:-$ROOT/checkpoints/q1_recovery_v3_boot_kneel_ppo.pt}"
# Prefer final model from latest full run if present.
FULL_DIR=$(ls -td "$ROOT"/logs/rsl_rl/q1_recovery_v3_boot_kneel/*boot_kneel_full* 2>/dev/null | head -1 || true)
if [[ -n "${FULL_DIR:-}" && -f "$FULL_DIR/model_4999.pt" ]]; then
  CKPT="$FULL_DIR/model_4999.pt"
elif [[ -n "${FULL_DIR:-}" ]]; then
  LATEST=$(ls -1 "$FULL_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
  [[ -n "${LATEST:-}" ]] && CKPT="$LATEST"
fi
OUT="${OUT:-$ROOT/docs/reference/review_recovery_v3/live_boot_kneel}"
STEPS="${STEPS:-1800}"
STOP_HOLD="${STOP_HOLD:-0.8}"
TAIL_S="${TAIL_S:-1.5}"
SEED="${SEED:-42}"

mkdir -p "$OUT"
echo "[record] ckpt=$CKPT out=$OUT seed=$SEED"
bash evaluate_recovery_v3.sh --checkpoint "$CKPT" --mode boot_kneel --seed "$SEED" \
  --num_envs 1 --video --steps "$STEPS" \
  --stop-after-stand-s "$STOP_HOLD" --tail-after-stand-s "$TAIL_S" \
  --out "$OUT"

MP4="$OUT/recovery_eval_boot_kneel.mp4"
test -f "$MP4"
python3 tools/upload_gofile.py "$MP4" "$OUT/recovery_eval_boot_kneel.gofile.json"
cp -f "$OUT/recovery_eval_boot_kneel.gofile.json" \
  docs/reference/review_recovery_v3/live_boot_kneel.gofile.json

python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT")
j = json.loads((out / "evaluation.json").read_text())
s = j["summary"]
g = json.loads((out / "recovery_eval_boot_kneel.gofile.json").read_text())
url = g.get("data", {}).get("downloadPage")
note = (
    f"seed={j.get('seed')} stand={s['kneel_then_stand']} hold={s['best_stand_hold_s']} "
    f"steps={s.get('ran_steps')} stopped={s.get('stopped_after_stand')}"
)
md = Path("docs/demos/LIVE_BOOT_KNEEL_UPLOADS.md")
md.write_text(
    "# Boot-kneel (assisted kneel→stand) demo upload\n\n"
    "| 動作 | PPO | gofile | notes |\n|---|---|---|---|\n"
    f"| 跪姿起身 boot_kneel | \`checkpoints/q1_recovery_v3_boot_kneel_ppo.pt\` | {url} | {note} |\n"
)
print("RESULT boot_kneel", note, "url", url)
print("wrote", md)
PY
echo "=== DONE $(date -Is) ==="
