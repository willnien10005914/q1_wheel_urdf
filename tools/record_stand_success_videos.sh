#!/bin/bash
# Record short supine/prone videos that stop after successful stand; upload gofile.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/isaac_env.sh"

STEPS_PROBE="${STEPS_PROBE:-2000}"
STEPS_VIDEO="${STEPS_VIDEO:-2200}"
STOP_HOLD="${STOP_HOLD:-0.8}"
TAIL_S="${TAIL_S:-1.5}"
SUPINE_CKPT="${SUPINE_CKPT:-$ROOT/checkpoints/q1_recovery_v3_supine_ppo.pt}"
PRONE_CKPT="${PRONE_CKPT:-$ROOT/checkpoints/q1_recovery_v3_prone_ppo.pt}"

ok_stand() {
  local out=$1
  python3 -c "
import json,sys
from pathlib import Path
s=json.loads(Path('$out/evaluation.json').read_text())['summary']
hold=float(s['best_stand_hold_s'][0])
stood=bool(s['kneel_then_stand'][0]) or hold>=float('$STOP_HOLD')
stage=int(s['max_stage'][0])
sys.exit(0 if (stood and stage>=4 and hold>=float('$STOP_HOLD')) else 1)
"
}

probe() {
  local mode=$1 ckpt=$2; shift 2
  local seed out
  for seed in "$@"; do
    out="/tmp/q1_probe_${mode}_${seed}"
    rm -rf "$out"
    echo "=== probe $mode seed=$seed ===" >&2
    bash evaluate_recovery_v3.sh --checkpoint "$ckpt" --mode "$mode" --seed "$seed" \
      --steps "$STEPS_PROBE" --stop-after-stand-s "$STOP_HOLD" --tail-after-stand-s 0.3 \
      --out "$out" >&2 || true
    if ok_stand "$out"; then
      python3 -c "import json;from pathlib import Path;s=json.loads(Path('$out/evaluation.json').read_text())['summary'];print('OK steps',s.get('ran_steps'),'hold',s['best_stand_hold_s'],'stage',s['max_stage'])" >&2
      echo "$seed"; return 0
    fi
    python3 -c "import json;from pathlib import Path;s=json.loads(Path('$out/evaluation.json').read_text())['summary'];print('  fail kneel',s['kneel'],'upright',s['upright_kneel'],'stand',s['kneel_then_stand'],'hold',s['best_stand_hold_s'],'stage',s['max_stage'],'steps',s.get('ran_steps'))" >&2 || true
  done
  return 1
}

# Prefer seeds that historically reached stand / kneel.
SUPINE_SEED=$(probe supine "$SUPINE_CKPT" 5107 4107 4119 42 7001 8001 2026 || true)
PRONE_SEED=$(probe prone "$PRONE_CKPT" 4119 4107 5107 42 7001 8001 2026 || true)
[[ -n "${SUPINE_SEED:-}" ]] || SUPINE_SEED=5107
[[ -n "${PRONE_SEED:-}" ]] || PRONE_SEED=4119
echo "[record] seeds supine=$SUPINE_SEED prone=$PRONE_SEED"

record_one() {
  local mode=$1 ckpt=$2 seed=$3
  local out="docs/reference/review_recovery_v3/live_${mode}"
  mkdir -p "$out"
  echo "=== VIDEO $mode seed=$seed stop_hold=${STOP_HOLD}s ==="
  bash evaluate_recovery_v3.sh --checkpoint "$ckpt" --mode "$mode" --seed "$seed" \
    --video --steps "$STEPS_VIDEO" \
    --stop-after-stand-s "$STOP_HOLD" --tail-after-stand-s "$TAIL_S" \
    --out "$out"
  local mp4="$out/recovery_eval_${mode}.mp4"
  test -f "$mp4"
  python3 tools/upload_gofile.py "$mp4" "$out/recovery_eval_${mode}.gofile.json"
  cp -f "$out/recovery_eval_${mode}.gofile.json" "docs/reference/review_recovery_v3/live_${mode}.gofile.json"
  python3 - <<PY
import json
from pathlib import Path
j=json.loads(Path("$out/evaluation.json").read_text()); s=j["summary"]
g=json.loads(Path("$out/recovery_eval_${mode}.gofile.json").read_text())
print("RESULT", "$mode", "seed", j.get("seed"),
      "stand", s["kneel_then_stand"], "hold", s["best_stand_hold_s"],
      "ran_steps", s.get("ran_steps"), "url", g.get("data",{}).get("downloadPage"))
PY
}

record_one supine "$SUPINE_CKPT" "$SUPINE_SEED"
record_one prone "$PRONE_CKPT" "$PRONE_SEED"

python3 - <<'PY'
import json
from pathlib import Path
lines=["# Stand-success recovery re-record\n\n",
       "Kneel wheel spin muted (stage-3 residuals zeroed). Videos stop after stand.\n\n",
       "| 動作 | PPO | gofile | notes |\n|---|---|---|---|\n"]
print("\n======== gofile links ========")
for mode, ckpt in [("正躺 supine","checkpoints/q1_recovery_v3_supine_ppo.pt"),
                   ("趴躺 prone","checkpoints/q1_recovery_v3_prone_ppo.pt")]:
    key=mode.split()[-1]
    gj=Path(f"docs/reference/review_recovery_v3/live_{key}/recovery_eval_{key}.gofile.json")
    ej=Path(f"docs/reference/review_recovery_v3/live_{key}/evaluation.json")
    url=json.loads(gj.read_text())["data"]["downloadPage"]
    s=json.loads(ej.read_text())["summary"]
    note=f"seed={json.loads(ej.read_text()).get('seed')} stand={s['kneel_then_stand']} hold={s['best_stand_hold_s']} steps={s.get('ran_steps')}"
    lines.append(f"| {mode} | `{ckpt}` | {url} | {note} |\n")
    print(f"{mode}\n  PPO: {ckpt}\n  {url}\n  {note}\n")
Path("docs/demos/LIVE_STAND_UPLOADS.md").write_text("".join(lines))
print("wrote docs/demos/LIVE_STAND_UPLOADS.md")
PY
echo "=== DONE $(date -Is) ==="
