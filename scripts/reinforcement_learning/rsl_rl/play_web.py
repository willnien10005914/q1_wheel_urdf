"""HTTP bridge between Isaac Sim play.py and the THREE.js joint UI.

Serves the repo as static files and a small JSON API so the browser can:
  * stream live joint pos/vel/torque and the PPO twist command
  * override individual motors (those targets are written into Isaac)
  * request a hard env reset
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class PlayWebState:
    """Thread-safe snapshot + incoming commands from the browser."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.snapshot: dict[str, Any] = {"ok": False, "joints": {}, "cmd": {"vx": 0.0, "vy": 0.0, "yaw": 0.0}}
        self.overrides: dict[str, float] = {}
        self.reset_requested = False
        self.release_all = False
        self.web_cmd: dict[str, float] | None = None
        self.pose_cmd: str | None = None

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            payload = dict(payload)
            payload["overrides"] = dict(self.overrides)
            self.snapshot = payload

    def snapshot_copy(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.snapshot))

    def apply_joint_msg(self, body: dict[str, Any]) -> None:
        with self._lock:
            for name in body.get("release") or []:
                self.overrides.pop(str(name), None)
            for name, value in (body.get("overrides") or {}).items():
                try:
                    self.overrides[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
            if body.get("release_all"):
                self.overrides.clear()

    def pop_overrides(self) -> dict[str, float]:
        with self._lock:
            if self.release_all:
                self.overrides.clear()
                self.release_all = False
            return dict(self.overrides)

    def request_reset(self) -> None:
        with self._lock:
            self.reset_requested = True
            self.overrides.clear()
            self.web_cmd = None
            self.pose_cmd = None

    def consume_reset(self) -> bool:
        with self._lock:
            flag = self.reset_requested
            self.reset_requested = False
            return flag

    def request_pose(self, name: str) -> None:
        with self._lock:
            self.pose_cmd = str(name)
            self.web_cmd = None

    def consume_pose(self) -> str | None:
        with self._lock:
            name = self.pose_cmd
            self.pose_cmd = None
            return name


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes]:
    raw = json.dumps(payload).encode("utf-8")
    return status, raw


class PlayWebHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".stl": "model/stl",
        ".urdf": "application/xml",
        ".json": "application/json",
    }
    state: PlayWebState

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if self.path.startswith("/api/"):
            return
        super().log_message(fmt, *args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            status, raw = _json_bytes(self.state.snapshot_copy())
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw_in = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw_in.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}
        if not isinstance(body, dict):
            body = {}

        if parsed.path == "/api/joints":
            self.state.apply_joint_msg(body)
            payload = {"ok": True, "overrides": self.state.pop_overrides()}
        elif parsed.path == "/api/release":
            self.state.apply_joint_msg({"release_all": True})
            payload = {"ok": True}
        elif parsed.path == "/api/reset":
            self.state.request_reset()
            payload = {"ok": True}
        elif parsed.path == "/api/pose":
            name = str(body.get("name") or "").strip().lower()
            if name in {"kneel", "stand", "slide", "skate"}:
                self.state.request_pose(name)
                payload = {"ok": True, "pose": name}
            else:
                status, raw = _json_bytes({"ok": False, "error": "name must be kneel, stand, slide or skate"}, 400)
                self.send_response(status)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
        elif parsed.path == "/api/command":
            cmd = {}
            for key in ("vx", "vy", "yaw"):
                if key in body:
                    try:
                        cmd[key] = float(body[key])
                    except (TypeError, ValueError):
                        pass
            with self.state._lock:
                self.state.web_cmd = cmd or None
            payload = {"ok": True, "cmd": cmd}
        else:
            status, raw = _json_bytes({"ok": False, "error": "unknown endpoint"}, 404)
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        status, raw = _json_bytes(payload)
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start_play_web(port: int = 8766, open_browser: bool = True) -> PlayWebState:
    state = PlayWebState()

    class BoundHandler(PlayWebHandler):
        pass

    BoundHandler.state = state
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(("127.0.0.1", port), BoundHandler)
    thread = threading.Thread(target=httpd.serve_forever, name="q1-play-web", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/web/"
    print(f"[INFO] Web motor UI: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"[WARN] Could not open browser: {exc}")
    return state
