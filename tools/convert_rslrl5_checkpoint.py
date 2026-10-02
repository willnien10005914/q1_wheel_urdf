#!/usr/bin/env python3
"""Convert rsl-rl <5 ActorCritic checkpoints (with obs normalizers) to rsl-rl 5 layout."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert(path: Path, force: bool = False) -> Path:
    out = path.with_name(path.stem + "_rslrl5" + path.suffix)
    if out.is_file() and not force:
        print(f"[skip] exists {out}")
        return out
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise SystemExit(f"unexpected checkpoint type: {type(loaded)}")
    if "actor_state_dict" in loaded:
        print(f"[skip] already new layout: {path}")
        return path
    if "model_state_dict" not in loaded:
        raise SystemExit(f"no model_state_dict in {path}")

    actor: dict = {}
    critic: dict = {}
    for key, value in loaded["model_state_dict"].items():
        if key == "std":
            actor["distribution.std_param"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor["obs_normalizer." + key[len("actor_obs_normalizer.") :]] = value
        elif key.startswith("critic_obs_normalizer."):
            critic["obs_normalizer." + key[len("critic_obs_normalizer.") :]] = value
        elif key.startswith("actor."):
            actor["mlp." + key[len("actor.") :]] = value
        elif key.startswith("critic."):
            critic["mlp." + key[len("critic.") :]] = value
        else:
            raise SystemExit(f"unrecognized key: {key}")

    converted = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": loaded.get("optimizer_state_dict"),
        "iter": loaded.get("iter", 0),
        "infos": loaded.get("infos"),
    }
    torch.save(converted, out)
    print(f"[ok] {path.name} -> {out.name} (actor={len(actor)} critic={len(critic)} tensors)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for p in args.checkpoints:
        convert(p, force=args.force)


if __name__ == "__main__":
    main()
