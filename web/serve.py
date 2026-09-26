#!/usr/bin/env python3
"""Serve the joint-control web UI from the URDF package root.

Also accepts POST /api/recording to write MediaPipe→web motor-angle recordings under
docs/reference/ for ``tools/apply_unbox_recording.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socketserver
import webbrowser
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "docs", "reference")


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

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        if path == "/api/recording":
            self._save_recording(body)
            return
        self.send_error(404, "unknown api")

    def _save_recording(self, body: dict) -> None:
        name = body.get("filename") or "unbox_web_recording.json"
        name = os.path.basename(name)
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.json", name):
            self.send_error(400, "bad filename")
            return
        os.makedirs(REF_DIR, exist_ok=True)
        out = os.path.join(REF_DIR, name)
        payload = body.get("recording")
        if not isinstance(payload, dict):
            self.send_error(400, "recording must be an object")
            return
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        data = json.dumps({"ok": True, "path": f"docs/reference/{name}"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheel Humanoid joint-control server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://127.0.0.1:{args.port}/web/"
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        print(f"Serving {ROOT}", flush=True)
        print(f"Open {url}", flush=True)
        print(f"POST /api/recording → {REF_DIR}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
