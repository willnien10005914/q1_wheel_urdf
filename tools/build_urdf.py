#!/usr/bin/env python3
"""Build urdf/wheel_humanoid.urdf from the structural URDF + config/q1_wheel_components.yaml.

The structural URDF (urdf/wheel_humanoid_structural.urdf) carries the CAD geometry with
uniform-density mass estimates. This script layers the hardware sheet on top:

  * adds the joints that exist on the BI2 slide but have no mesh yet (neck, wrists, grippers),
  * adds the Xsens MTi-630 as `imu_link` (fixed joint on the pelvis at hip-axis height),
  * adds every CubeMars stator as a point mass on the PARENT link at the joint origin
    (parallel-axis theorem, CoM/inertia recomputed),
  * rescales the structural masses so the total mass hits `robot.target_total_mass_kg`,
  * writes joint effort/velocity limits from the motor SKU (slide peak torque, no-load speed),
  * zeroes URDF joint friction/damping: friction is modelled inside the Isaac Lab
    CubeMars actuator (do not double count it in PhysX),
  * sets the wheel collision cylinder radius from `robot.wheel_radius_m`.

Usage:
    python tools/build_urdf.py            # writes urdf/wheel_humanoid.urdf
    python tools/build_urdf.py --check    # print the mass budget only
"""
from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STRUCTURAL = os.path.join(ROOT, "urdf", "wheel_humanoid_structural.urdf")
OUTPUT = os.path.join(ROOT, "urdf", "wheel_humanoid.urdf")
CONFIG = os.path.join(ROOT, "config", "q1_wheel_components.yaml")


def _fmt(v: float) -> str:
    return f"{v:.6e}" if abs(v) < 1e-3 else f"{v:.6f}"


def _read_inertial(link: ET.Element):
    ine = link.find("inertial")
    m = float(ine.find("mass").get("value"))
    c = np.array([float(x) for x in ine.find("origin").get("xyz").split()])
    I = ine.find("inertia")
    T = np.array(
        [
            [float(I.get("ixx")), float(I.get("ixy")), float(I.get("ixz"))],
            [float(I.get("ixy")), float(I.get("iyy")), float(I.get("iyz"))],
            [float(I.get("ixz")), float(I.get("iyz")), float(I.get("izz"))],
        ]
    )
    return m, c, T


def _write_inertial(link: ET.Element, m: float, c: np.ndarray, T: np.ndarray):
    ine = link.find("inertial")
    ine.find("mass").set("value", f"{m:.4f}")
    ine.find("origin").set("xyz", " ".join(_fmt(x) for x in c))
    I = ine.find("inertia")
    I.set("ixx", _fmt(T[0, 0]))
    I.set("ixy", _fmt(T[0, 1]))
    I.set("ixz", _fmt(T[0, 2]))
    I.set("iyy", _fmt(T[1, 1]))
    I.set("iyz", _fmt(T[1, 2]))
    I.set("izz", _fmt(T[2, 2]))


def _shift(T: np.ndarray, m: float, d: np.ndarray) -> np.ndarray:
    """Parallel-axis shift of an inertia tensor about the CoM by vector d."""
    return T + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def add_point_mass(link: ET.Element, m_add: float, p: np.ndarray, I_add: np.ndarray | None = None):
    """Combine the link inertial with a point (or small body) mass at position p (link frame)."""
    m0, c0, T0 = _read_inertial(link)
    if I_add is None:
        I_add = np.zeros((3, 3))
    m = m0 + m_add
    c = (m0 * c0 + m_add * p) / m
    T = _shift(T0, m0, c0 - c) + _shift(I_add, m_add, p - c)
    _write_inertial(link, m, c, T)


def scale_inertial(link: ET.Element, k: float):
    m0, c0, T0 = _read_inertial(link)
    _write_inertial(link, m0 * k, c0, T0 * k)


def cylinder_inertia(m: float, r: float, h: float, axis: str = "z") -> np.ndarray:
    i_axial = 0.5 * m * r * r
    i_rad = m * (3 * r * r + h * h) / 12.0
    d = {"x": (i_axial, i_rad, i_rad), "y": (i_rad, i_axial, i_rad), "z": (i_rad, i_rad, i_axial)}[axis]
    return np.diag(d)


def make_link(name: str, mass: float, vis: dict | None, material: str) -> ET.Element:
    link = ET.Element("link", name=name)
    ine = ET.SubElement(link, "inertial")
    ET.SubElement(ine, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ine, "mass", value=f"{mass:.4f}")
    if vis:
        r, h = vis["radius"], vis["length"]
        rpy = vis.get("rpy", [0, 0, 0])
        axis = "y" if abs(rpy[0]) > 1.0 else "z"
        T = cylinder_inertia(mass, r, h, axis)
    else:
        T = np.eye(3) * 1e-5
    ET.SubElement(
        ine,
        "inertia",
        ixx=_fmt(T[0, 0]),
        ixy="0",
        ixz="0",
        iyy=_fmt(T[1, 1]),
        iyz="0",
        izz=_fmt(T[2, 2]),
    )
    if vis:
        rpy_s = " ".join(str(x) for x in vis.get("rpy", [0, 0, 0]))
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            ET.SubElement(el, "origin", xyz="0 0 0", rpy=rpy_s)
            geo = ET.SubElement(el, "geometry")
            ET.SubElement(geo, "cylinder", radius=str(vis["radius"]), length=str(vis["length"]))
            if tag == "visual":
                ET.SubElement(el, "material", name=material)
    return link


def make_joint(name: str, jtype: str, parent: str, child: str, xyz, rpy, axis=None, limit=None) -> ET.Element:
    j = ET.Element("joint", name=name, type=jtype)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    ET.SubElement(j, "origin", xyz=" ".join(str(x) for x in xyz), rpy=" ".join(str(x) for x in rpy))
    if axis is not None:
        ET.SubElement(j, "axis", xyz=" ".join(str(x) for x in axis))
    if limit is not None:
        ET.SubElement(j, "limit", **{k: str(v) for k, v in limit.items()})
    if jtype != "fixed":
        ET.SubElement(j, "dynamics", damping="0.0", friction="0.0")
    return j


def indent(elem: ET.Element, level: int = 0):
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only print the mass budget")
    ap.add_argument("--out", default=OUTPUT)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG))
    tree = ET.parse(STRUCTURAL)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}

    motors = cfg["motors"]
    jmap = cfg["joints"]
    extra = cfg.get("extra_joints", {})
    imu = cfg["imu"]
    robot = cfg["robot"]

    # ---- 1. extra joints/links from the slide (neck, wrists, grippers) -----------------------
    for jname, spec in extra.items():
        if jname in joints:
            continue
        vis = spec.get("visual_cylinder")
        link = make_link(spec["child"], spec["child_mass_kg"], vis, "dark")
        root.append(link)
        links[spec["child"]] = link
        motor = motors[jmap[jname]["motor"]]
        limit = {
            "lower": spec["lower"],
            "upper": spec["upper"],
            "effort": motor["tau_peak_slide_nm"],
            "velocity": round(motor["no_load_speed_rpm"] * 2 * math.pi / 60.0, 3),
        }
        j = make_joint(jname, "revolute", spec["parent"], spec["child"], spec["xyz"], spec["rpy"], spec["axis"], limit)
        root.append(j)
        joints[jname] = j

    # ---- 2. IMU link --------------------------------------------------------------------------
    if "imu_link" not in links:
        sx, sy, sz = imu["size_m"]
        m = imu["mass_kg"]
        link = ET.Element("link", name="imu_link")
        ine = ET.SubElement(link, "inertial")
        ET.SubElement(ine, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ine, "mass", value=f"{m:.4f}")
        ET.SubElement(
            ine,
            "inertia",
            ixx=_fmt(m * (sy * sy + sz * sz) / 12),
            ixy="0",
            ixz="0",
            iyy=_fmt(m * (sx * sx + sz * sz) / 12),
            iyz="0",
            izz=_fmt(m * (sx * sx + sy * sy) / 12),
        )
        v = ET.SubElement(link, "visual")
        ET.SubElement(v, "origin", xyz="0 0 0", rpy="0 0 0")
        geo = ET.SubElement(v, "geometry")
        ET.SubElement(geo, "box", size=f"{sx} {sy} {sz}")
        ET.SubElement(v, "material", name="imu_orange")
        root.append(link)
        links["imu_link"] = link
        j = make_joint("imu_joint", "fixed", imu["parent_link"], "imu_link", imu["xyz"], imu["rpy"])
        root.append(j)
        joints["imu_joint"] = j
        mat = ET.Element("material", name="imu_orange")
        ET.SubElement(mat, "color", rgba="0.95 0.45 0.05 1")
        root.insert(3, mat)

    # ---- 3. mass budget -----------------------------------------------------------------------
    structural_links = [n for n in links if n not in {spec["child"] for spec in extra.values()} and n != "imu_link"]
    m_struct = sum(_read_inertial(links[n])[0] for n in structural_links)
    m_motors = sum(motors[jmap[j]["motor"]]["mass_kg"] for j in jmap)
    m_extra = sum(spec["child_mass_kg"] for spec in extra.values())
    m_imu = imu["mass_kg"]
    target = robot["target_total_mass_kg"]
    k = (target - m_motors - m_extra - m_imu) / m_struct
    print(f"structural (uniform density) : {m_struct:8.3f} kg  -> x{k:.4f}")
    print(f"CubeMars stators (24)        : {m_motors:8.3f} kg")
    print(f"slide-only links (5)         : {m_extra:8.3f} kg")
    print(f"IMU                          : {m_imu:8.3f} kg")
    print(f"target total                 : {target:8.3f} kg")
    if args.check:
        return
    for n in structural_links:
        scale_inertial(links[n], k)

    # ---- 4. motor stators on the parent link at the joint origin ------------------------------
    for jname, spec in jmap.items():
        j = joints[jname]
        motor = motors[spec["motor"]]
        parent = j.find("parent").get("link")
        origin = np.array([float(x) for x in j.find("origin").get("xyz").split()])
        r = motor["diameter_mm"] / 2000.0
        h = motor["length_mm"] / 1000.0
        axis = np.array([float(x) for x in j.find("axis").get("xyz").split()])
        ax = "xyz"[int(np.argmax(np.abs(axis)))]
        add_point_mass(links[parent], motor["mass_kg"], origin, cylinder_inertia(motor["mass_kg"], r, h, ax))
        # joint limits from the SKU; PhysX friction/damping are zero (handled by the actuator model)
        lim = j.find("limit")
        lim.set("effort", str(motor["tau_peak_slide_nm"]))
        lim.set("velocity", str(round(motor["no_load_speed_rpm"] * 2 * math.pi / 60.0, 3)))
        dyn = j.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(j, "dynamics")
        dyn.set("damping", "0.0")
        dyn.set("friction", "0.0")

    # ---- 5. wheel collision radius ------------------------------------------------------------
    for wl in ("l_wheel_link", "r_wheel_link"):
        cyl = links[wl].find("collision/geometry/cylinder")
        cyl.set("radius", str(robot["wheel_radius_m"]))
        cyl.set("length", str(robot["wheel_width_m"]))

    total = sum(_read_inertial(l)[0] for l in links.values())
    print(f"final URDF total mass        : {total:8.3f} kg over {len(links)} links, {len(joints)} joints")

    # ---- 6. header comment + write -------------------------------------------------------------
    header = ET.Comment(
        " GENERATED by tools/build_urdf.py from urdf/wheel_humanoid_structural.urdf + "
        "config/q1_wheel_components.yaml. Do not edit by hand.\n"
        f"     Q1 Wheel / BI2 Wheel: {total:.2f} kg, 24 CubeMars joints "
        "(AKE90-8 hip/knee, AK10-9 torso/shoulder/elbow/wheel, AK45-36 neck, AK45-10 wrist/gripper),\n"
        f"     Xsens MTi-630 on pelvis at hip-axis height, wheel radius {robot['wheel_radius_m']} m, "
        f"CAN {cfg['bus']['can_rate_hz']} Hz. Joint friction/damping = 0: modelled in the Isaac Lab actuator. "
    )
    root.insert(0, header)
    indent(root)
    tree.write(args.out, xml_declaration=True, encoding="utf-8")
    print(f"wrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
