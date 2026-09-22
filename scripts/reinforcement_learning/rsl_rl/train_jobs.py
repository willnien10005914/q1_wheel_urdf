"""Start / poll headless PPO jobs (posture, slide) from the web UI or CLI.

Training needs the GPU. Play (Isaac Sim) and train cannot share it: if play.py is
running, start() returns an error telling the user to ./stop_isaac.sh first.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

JOBS: dict[str, dict[str, str]] = {
    "posture": {
        "script": "train_posture.sh",
        "log": "logs/train_posture.log",
        "pidfile": "logs/train_posture.pid",
        "ckpt": "checkpoints/q1_posture_ppo.pt",
        "label": "kneel/stand PPO",
    },
    "slide": {
        "script": "train_slide.sh",
        "log": "logs/train_slide.log",
        "pidfile": "logs/train_slide.pid",
        "ckpt": "checkpoints/q1_slide_ppo.pt",
        "label": "slide / X2-skate PPO",
    },
}


def _path(rel: str) -> str:
    return os.path.join(ROOT, rel)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(pidfile: str) -> int | None:
    try:
        raw = open(pidfile, encoding="utf-8").read().strip()
        pid = int(raw)
    except (OSError, ValueError):
        return None
    if not _pid_alive(pid):
        return None
    return pid


def play_is_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", f"{ROOT}/scripts/reinforcement_learning/rsl_rl/play.py"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out.strip())


def _any_train_running() -> str | None:
    for name, spec in JOBS.items():
        if _read_pid(_path(spec["pidfile"])) is not None:
            return name
    try:
        out = subprocess.check_output(["pgrep", "-f", f"{ROOT}/scripts/reinforcement_learning/rsl_rl/train.py"], text=True)
        if out.strip():
            return "train"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def _log_tail(log_path: str, n: int = 8) -> str:
    try:
        lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def status(name: str | None = None) -> dict[str, Any]:
    names = [name] if name in JOBS else list(JOBS)
    out: dict[str, Any] = {}
    for key in names:
        spec = JOBS[key]
        pid = _read_pid(_path(spec["pidfile"]))
        ckpt = _path(spec["ckpt"])
        item = {
            "name": key,
            "label": spec["label"],
            "running": pid is not None,
            "pid": pid,
            "checkpoint": os.path.isfile(ckpt),
            "checkpoint_mtime": os.path.getmtime(ckpt) if os.path.isfile(ckpt) else None,
            "log": _log_tail(_path(spec["log"]), 6),
        }
        out[key] = item
    if name in JOBS:
        return out[name]
    return {"jobs": out, "play_running": play_is_running()}


def start(name: str) -> dict[str, Any]:
    if name not in JOBS:
        return {"ok": False, "error": "name must be posture or slide"}
    busy = _any_train_running()
    if busy:
        return {"ok": False, "error": f"training already running ({busy}). Wait or ./stop_isaac.sh"}
    if play_is_running():
        return {
            "ok": False,
            "error": "Isaac Sim play is using the GPU. Stop it with ./stop_isaac.sh, then click Train again.",
        }
    spec = JOBS[name]
    os.makedirs(_path("logs"), exist_ok=True)
    log_path = _path(spec["log"])
    script = _path(spec["script"])
    log_f = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        ["bash", script],
        cwd=ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    with open(_path(spec["pidfile"]), "w", encoding="utf-8") as fh:
        fh.write(str(proc.pid))
    return {"ok": True, "name": name, "pid": proc.pid, "log": spec["log"], "label": spec["label"]}


def stop(name: str | None = None) -> dict[str, Any]:
    names = [name] if name in JOBS else list(JOBS)
    stopped: list[int] = []
    for key in names:
        pid = _read_pid(_path(JOBS[key]["pidfile"]))
        if pid is None:
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        stopped.append(pid)
        time.sleep(0.2)
    return {"ok": True, "stopped": stopped}
