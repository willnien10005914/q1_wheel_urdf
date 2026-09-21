#!/bin/bash
# Stop every Isaac Sim / Isaac Lab / motor-web process started from this repo
# and free GPU, Kit, and HTTP ports so play/train can start cleanly.
#
#   ./stop_isaac.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_PORTS=(8765 8766)

log() { printf '[stop] %s\n' "$*"; }

collect_matches() {
  local pattern="$1"
  pgrep -f -- "$pattern" 2>/dev/null || true
}

children_of() {
  local pid="$1"
  local kid
  for kid in $(pgrep -P "$pid" 2>/dev/null || true); do
    children_of "$kid"
    printf '%s\n' "$kid"
  done
}

pids_on_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true
    return
  fi
  ss -lptn "sport = :$port" 2>/dev/null \
    | grep -oE 'pid=[0-9]+' \
    | cut -d= -f2 \
    | sort -u || true
}

declare -A SEEN=()
PIDS=()

add_pid() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && return 0
  [[ -n "${SEEN[$pid]:-}" ]] && return 0
  SEEN[$pid]=1
  PIDS+=("$pid")
}

add_tree() {
  local pid="$1"
  local child
  add_pid "$pid"
  while read -r child; do
    [[ -n "$child" ]] && add_pid "$child"
  done < <(children_of "$pid")
}

# Repo entry points (play, train, record, tests, standalone web).
PATTERNS=(
  "$ROOT/scripts/reinforcement_learning/rsl_rl/play.py"
  "$ROOT/scripts/reinforcement_learning/rsl_rl/train.py"
  "$ROOT/web/serve.py"
  "$ROOT/tests/"
  "$ROOT/play_skateboard.sh"
  "$ROOT/train_skate.sh"
  "$ROOT/train_skateboard.sh"
  "$ROOT/record_skate.sh"
  "$ROOT/record_stand_skate.sh"
  "$ROOT/run_isaac.sh"
)

for pattern in "${PATTERNS[@]}"; do
  for pid in $(collect_matches "$pattern"); do
    add_tree "$pid"
  done
done

for port in "${WEB_PORTS[@]}"; do
  for pid in $(pids_on_port "$port"); do
    add_tree "$pid"
  done
done

# Kit / telemetry leftovers whose command line still points at this checkout.
for pid in $(collect_matches "omni.telemetry.transmitter"); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$ROOT"; then
    add_tree "$pid"
  fi
done

if ((${#PIDS[@]} == 0)); then
  log "Nothing running."
  exit 0
fi

log "Stopping ${#PIDS[@]} process(es): ${PIDS[*]}"
kill -TERM "${PIDS[@]}" 2>/dev/null || true

deadline=$((SECONDS + 8))
while ((SECONDS < deadline)); do
  alive=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  ((${#alive[@]} == 0)) && break
  sleep 0.4
done

leftover=()
for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    leftover+=("$pid")
  fi
done
if ((${#leftover[@]} > 0)); then
  log "Force killing: ${leftover[*]}"
  kill -KILL "${leftover[@]}" 2>/dev/null || true
  sleep 0.3
fi

for port in "${WEB_PORTS[@]}"; do
  still=$(pids_on_port "$port")
  if [[ -n "$still" ]]; then
    log "Freeing port $port (pids $still)"
    kill -KILL $still 2>/dev/null || true
  fi
done

log "Isaac Sim / Lab / web UI stopped. GPU and ports 8765/8766 are free."
log "Start again with: ./play_skateboard.sh   or   python web/serve.py"
