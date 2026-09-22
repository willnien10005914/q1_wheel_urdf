#!/usr/bin/env python3
"""Serve the joint-control web UI from the URDF package root.

Also exposes POST/GET /api/train so kneel/stand and slide PPO jobs can be started
without Isaac Sim play occupying the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reinforcement_learning", "rsl_rl"))
import train_jobs  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".stl": "model/stl",
        ".urdf": "application/xml",
        ".json": "application/json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/train":
            self._json({"ok": True, **train_jobs.status()})
            return
        if parsed.path == "/api/state":
            self._json({"ok": False, "train": train_jobs.status(), "joints": {}, "cmd": {}})
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
        if parsed.path != "/api/train":
            self._json({"ok": False, "error": "unknown endpoint"}, 404)
            return
        name = str(body.get("name") or "").strip().lower()
        action = str(body.get("action") or "start").strip().lower()
        if action == "status":
            self._json({"ok": True, **train_jobs.status()})
        elif action == "stop":
            self._json(train_jobs.stop(name or None))
        else:
            self._json(train_jobs.start(name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheel Humanoid joint-control server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ThreadingHTTPServer.allow_reuse_address = True
    url = f"http://127.0.0.1:{args.port}/web/"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving {ROOT}", flush=True)
    print(f"Open {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
