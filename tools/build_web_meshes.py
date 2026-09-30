#!/usr/bin/env python3
"""Build lightweight visual meshes + URDF for the web joint UI."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import trimesh


def simplify_to(src: Path, dst: Path, face_target: int) -> None:
    mesh = trimesh.load(src, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    n0 = len(mesh.faces)
    if n0 > face_target:
        mesh = mesh.simplify_quadric_decimation(face_count=face_target)
    mesh.export(dst)
    print(f"{src.name}: {n0} -> {len(mesh.faces)} faces, {dst.stat().st_size // 1024} KB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--faces", type=int, default=5000)
    args = p.parse_args()
    root: Path = args.root
    src_urdf = root / "urdf" / "wheel_humanoid.urdf"
    out_urdf = root / "urdf" / "wheel_humanoid_web.urdf"
    out_mesh = root / "meshes" / "web"
    out_mesh.mkdir(parents=True, exist_ok=True)

    text = src_urdf.read_text()

    def repl(match: re.Match[str]) -> str:
        rel = match.group(1)
        name = Path(rel).name
        src = (root / "urdf" / rel).resolve()
        dst = out_mesh / name
        simplify_to(src, dst, args.faces)
        return f'filename="../meshes/web/{name}"'

    new = re.sub(r'filename="(\.\./meshes/visual/[^"]+\.stl)"', repl, text)
    new = re.sub(r"\s*<collision>.*?</collision>", "", new, flags=re.S)
    out_urdf.write_text(new)
    total = sum(f.stat().st_size for f in out_mesh.glob("*.stl"))
    print(f"wrote {out_urdf}  web meshes={total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
