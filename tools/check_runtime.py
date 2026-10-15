#!/usr/bin/env python3
"""Report actual CUDA kernel support and statically audit manager task registrations.

This does not import Isaac Sim or start a simulation. Run with the same Python
interpreter used by training; --require-cuda makes a missing/broken GPU fatal.
"""

import argparse
import ast
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _literal(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.Attribute) and node.attr == "__name__":
        return _literal(node.value, names)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(value, names) for value in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k, names): _literal(v, names) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.JoinedStr):
        return "".join(str(_literal(value.value, names)) if isinstance(value, ast.FormattedValue)
                       else value.value for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left, names) + _literal(node.right, names)
    raise ValueError(f"Unsupported registration expression: {ast.dump(node)}")


def manager_registrations(root):
    """Read registration declarations without executing any repository code."""
    source = Path(root) / "source/wheel_humanoid_lab"
    task_root = source / "wheel_humanoid_lab/tasks/manager_based"
    rows = []

    def visit(statements, names, path):
        for statement in statements:
            if isinstance(statement, ast.ImportFrom):
                package = names["__name__"].split(".")
                if statement.level:
                    base = package[:len(package) - statement.level + 1]
                    if statement.module:
                        base += statement.module.split(".")
                else:
                    base = statement.module.split(".") if statement.module else []
                for alias in statement.names:
                    names[alias.asname or alias.name] = ".".join(base + [alias.name])
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        names[target.id] = _literal(statement.value, names)
                    except (ValueError, KeyError, TypeError):
                        pass
            elif isinstance(statement, ast.For):
                for values in _literal(statement.iter, names):
                    local = dict(names)
                    targets = statement.target.elts if isinstance(statement.target, ast.Tuple) else [statement.target]
                    values = values if isinstance(statement.target, ast.Tuple) else [values]
                    local.update({target.id: value for target, value in zip(targets, values)})
                    visit(statement.body, local, path)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == "register":
                    values = {kw.arg: _literal(kw.value, names) for kw in call.keywords}
                    config = values.get("kwargs", {})
                    rows.append({"task": values["id"], "entry_point": values.get("entry_point"),
                                 "env_config": config.get("env_cfg_entry_point"),
                                 "agent_config": config.get("rsl_rl_cfg_entry_point"),
                                 "source": str(path.relative_to(root))})

    for path in sorted(task_root.glob("*/__init__.py")):
        module = ".".join(path.parent.relative_to(source).parts)
        visit(ast.parse(path.read_text()).body, {"__name__": module}, path)
    if not rows:
        raise ValueError(f"No manager task registrations found beneath {task_root}")
    invalid = [row for row in rows if row["entry_point"] != "isaaclab.envs:ManagerBasedRLEnv"
               or not row["env_config"] or not row["agent_config"]]
    if invalid:
        raise ValueError(f"Invalid manager registration declarations: {invalid}")
    return rows


def check_runtime(root=ROOT, require_cuda=False):
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "project_root": str(Path(root).resolve()),
              "packages": {}, "errors": [], "cuda_devices": []}
    for package in ("torch", "isaacsim", "isaaclab", "rsl-rl-lib"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        report["manager_registrations"] = manager_registrations(Path(root))
    except Exception as error:
        report["errors"].append(f"Manager registry audit: {error}")
    try:
        import torch

        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_arches"] = torch.cuda.get_arch_list() if report["cuda_available"] else []
        if require_cuda and not report["cuda_available"]:
            report["errors"].append("CUDA is unavailable in this Python environment")
        for index in range(torch.cuda.device_count()):
            row = {"index": index, "name": torch.cuda.get_device_name(index),
                   "compute_capability": list(torch.cuda.get_device_capability(index))}
            try:
                device = f"cuda:{index}"
                # Exercise PyTorch pointwise/reduction kernels and cuBLAS. A GPU
                # name or architecture allowlist cannot establish compatibility.
                value = torch.full((32, 32), 0.125, device=device)
                product = value @ value
                pointwise = torch.sin(value) + value.square()
                torch.cuda.synchronize(index)
                row["math_finite"] = bool(torch.isfinite(product).all() and torch.isfinite(pointwise).all())
                row["matmul_correct"] = bool(torch.allclose(product, torch.full_like(product, 0.5)))
                row["math_pass"] = row["math_finite"] and row["matmul_correct"]
                if not row["math_pass"]:
                    report["errors"].append(f"CUDA math verification failed on device {index}")
            except Exception as error:
                row["math_pass"] = False
                row["error"] = str(error)
                report["errors"].append(f"CUDA device {index}: {error}")
            report["cuda_devices"].append(row)
    except ImportError as error:
        report["cuda_available"] = False
        report["errors"].append(f"Torch import: {error}")
    report["pass"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check_runtime(args.project_root.resolve(), args.require_cuda)
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
