"""Configuration for the CubeMars quasi-direct-drive actuator model."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils import configclass

from .cubemars import CubeMarsActuator


@configclass
class CubeMarsActuatorCfg(ActuatorBaseCfg):
    """CubeMars AK/AKE actuator: firmware PD (position) or velocity loop in front of a voltage- and
    current-limited BLDC plant with an in-actuator friction budget, command delay and bus-voltage sag.

    All torque/speed quantities are on the OUTPUT (joint) side unless the name says ``motor``.
    """

    class_type: type = CubeMarsActuator

    sku: str = MISSING
    """CubeMars SKU string, for logging only (e.g. ``AKE90-8_KV35``)."""

    control_mode: Literal["position", "velocity"] = "position"
    """What the policy commands. ``position``: q_des with PD (stiffness/damping).
    ``velocity``: qd_des with a velocity loop using ``damping`` as the gain (wheels)."""

    # ---- electrical (motor side numbers from the datasheet) ----
    bus_voltage: float = 48.0
    bus_voltage_range: tuple[float, float] | None = None
    """Per-env bus voltage domain randomization, sampled fresh (restore-then-sample) on every reset."""
    kv_rpm_per_v: float = MISSING
    kt_nm_per_a: float = MISSING
    phase_resistance: float = MISSING
    """Phase-to-phase resistance in ohm."""
    gear_ratio: float = MISSING
    gear_efficiency: float = 0.90
    peak_current: float = MISSING
    """Driver peak current limit [A]."""
    cont_current: float | None = None
    """Rated (continuous) current [A]; only used for the saturation/thermal diagnostics."""
    peak_torque: float = MISSING
    """Operating peak output torque [Nm] (BI2 slide value). Also the effort clip."""
    cont_torque: float | None = None
    """Rated output torque [Nm], for the diagnostics."""

    # ---- friction budget (output side). PhysX joint friction MUST be 0 when these are used. ----
    coulomb_friction: float = 0.0
    """Coulomb friction torque [Nm]."""
    stribeck_friction: float = 0.0
    """Additional break-away torque [Nm] that decays with speed (Stribeck)."""
    stribeck_velocity: float = 0.1
    """Speed scale of the Stribeck term [rad/s]."""
    viscous_friction: float = 0.0
    """Viscous friction [Nm/(rad/s)]."""
    load_dep_friction: float = 0.0
    """Load dependent friction: fraction of |tau_motor| added to the Coulomb term (gearbox meshing)."""
    friction_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-env multiplicative DR on the whole friction budget (restore-then-sample)."""

    # ---- command path ----
    min_delay: int = 0
    max_delay: int = 0
    """Command delay in physics steps (CAN at 200 Hz => 1 step = 5 ms). Per env, resampled on reset."""

    # ---- power ----
    voltage_sag_gain: float = 0.0
    """Bus voltage drop [V] per Nm of total |applied torque| in this group (pack IR + wiring)."""
    voltage_min: float = 0.0
    """Floor of the sagged voltage [V]."""
