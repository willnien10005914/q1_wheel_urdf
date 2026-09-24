#!/usr/bin/env python3
"""Serve the joint-control web UI from the URDF package root."""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".stl": "model/stl",
        ".urdf": "application/xml",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)


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
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
