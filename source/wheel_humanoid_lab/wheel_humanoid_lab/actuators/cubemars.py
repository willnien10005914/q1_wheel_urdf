"""CubeMars quasi-direct-drive actuator model for Isaac Lab (explicit actuator).

This is the "BAM, but CubeMars" plant: the policy talks to a firmware loop (position PD or a
velocity loop), the loop asks for a torque, and the torque is what a voltage/current limited FOC
BLDC through a planetary gearbox can actually deliver at the current speed, minus a friction
budget that lives here (not in PhysX).

    tau_des   = kp (q_d - q) + kd (qd_d - qd) + tau_ff             (position mode)
              = kd (qd_d - qd) + tau_ff                            (velocity mode, wheels)
    V_eff     = clip(V_bus - k_sag * sum|tau_applied|, V_min, V_bus)
    e         = Ke * G * qd                                        (back-EMF, motor side)
    i_avail   = [(-V_eff - e) / R, (V_eff - e) / R]  ∩  [-I_peak, I_peak]
    i         = clip(tau_des / (Kt G eta), i_avail)
    tau_motor = Kt G eta i
    tau_fric  = [tau_c + tau_s exp(-(qd/v_s)^2)] tanh(qd/eps) (1 + k_load |tau_motor| / tau_peak) + b qd
    tau_out   = clip(tau_motor - s_fric * tau_fric, -tau_peak, tau_peak)

Per-env domain randomization (bus voltage, friction scale, command delay) is *restore-then-sample*:
every reset draws a fresh value from the nominal range, nothing accumulates across episodes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ActuatorBase
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .cubemars_cfg import CubeMarsActuatorCfg


class CubeMarsActuator(ActuatorBase):
    """Voltage/current limited CubeMars BLDC + gearbox with an in-actuator friction budget."""

    cfg: CubeMarsActuatorCfg

    def __init__(self, cfg: CubeMarsActuatorCfg, *args, **kwargs):
        # friction is modelled here -> make sure PhysX does not add its own
        cfg.friction = 0.0
        cfg.dynamic_friction = 0.0
        cfg.viscous_friction = 0.0
        # the plant clips the torque; the solver limit stays large (default for explicit actuators)
        if cfg.effort_limit is None:
            cfg.effort_limit = cfg.peak_torque
        if cfg.velocity_limit is None:
            cfg.velocity_limit = cfg.kv_rpm_per_v * cfg.bus_voltage * 2.0 * math.pi / 60.0 / cfg.gear_ratio
        super().__init__(cfg, *args, **kwargs)

        n, m, dev = self._num_envs, self.num_joints, self._device
        # motor constants (SI). Ke [V s/rad] == Kt [Nm/A] for a consistent SI machine; we use the
        # datasheet Kv for the back-EMF so the no-load speed matches the sheet.
        self._kt_out = cfg.kt_nm_per_a * cfg.gear_ratio * cfg.gear_efficiency
        self._ke_motor = 60.0 / (2.0 * math.pi * cfg.kv_rpm_per_v)
        self._R = cfg.phase_resistance
        self._G = cfg.gear_ratio
        self._i_peak = cfg.peak_current
        self._tau_peak = cfg.peak_torque
        self._tau_cont = cfg.cont_torque if cfg.cont_torque is not None else 0.45 * cfg.peak_torque

        # per-env DR state
        self.bus_voltage = torch.full((n, 1), cfg.bus_voltage, device=dev)
        self.friction_scale = torch.ones((n, 1), device=dev)
        self._sag_prev = torch.zeros((n, 1), device=dev)

        # command delay buffers (physics steps)
        self._delay = max(int(cfg.max_delay), 0)
        self._pos_buf = DelayBuffer(self._delay, n, device=dev)
        self._vel_buf = DelayBuffer(self._delay, n, device=dev)
        self._eff_buf = DelayBuffer(self._delay, n, device=dev)
        self._all_ids = torch.arange(n, dtype=torch.long, device=dev)

        # diagnostics
        self.current = torch.zeros((n, m), device=dev)
        self.motor_torque = torch.zeros((n, m), device=dev)
        self.friction_torque = torch.zeros((n, m), device=dev)
        self.saturated = torch.zeros((n, m), device=dev, dtype=torch.bool)
        self.voltage_limited = torch.zeros((n, m), device=dev, dtype=torch.bool)

        self.reset(slice(None))

    # ------------------------------------------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | slice):
        if env_ids is None or env_ids == slice(None):
            ids = self._all_ids
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(self._device, dtype=torch.long)
        else:
            ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self._device)
        k = ids.numel()
        # restore-then-sample: values are drawn from the nominal range, never multiplied onto
        # the previous episode's value.
        if self.cfg.bus_voltage_range is not None:
            lo, hi = self.cfg.bus_voltage_range
            self.bus_voltage[ids, 0] = torch.empty(k, device=self._device).uniform_(lo, hi)
        else:
            self.bus_voltage[ids, 0] = self.cfg.bus_voltage
        lo, hi = self.cfg.friction_scale_range
        self.friction_scale[ids, 0] = torch.empty(k, device=self._device).uniform_(lo, hi)
        self._sag_prev[ids] = 0.0
        if self._delay > 0:
            lags = torch.randint(int(self.cfg.min_delay), self._delay + 1, (k,), dtype=torch.int, device=self._device)
            for buf in (self._pos_buf, self._vel_buf, self._eff_buf):
                buf.set_time_lag(lags, ids)
                buf.reset(ids)

    # ------------------------------------------------------------------------------------------
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        q_d = control_action.joint_positions
        qd_d = control_action.joint_velocities
        tau_ff = control_action.joint_efforts
        if self._delay > 0:
            q_d = self._pos_buf.compute(q_d)
            qd_d = self._vel_buf.compute(qd_d)
            tau_ff = self._eff_buf.compute(tau_ff)

        # ---- firmware loop -------------------------------------------------------------------
        if self.cfg.control_mode == "position":
            tau_des = self.stiffness * (q_d - joint_pos) + self.damping * (qd_d - joint_vel) + tau_ff
        else:
            tau_des = self.damping * (qd_d - joint_vel) + tau_ff
        self.computed_effort = tau_des

        # ---- electrical plant ----------------------------------------------------------------
        v_eff = torch.clamp(
            self.bus_voltage - self.cfg.voltage_sag_gain * self._sag_prev, min=self.cfg.voltage_min
        ).clamp(max=self.bus_voltage)
        back_emf = self._ke_motor * self._G * joint_vel
        i_hi = torch.minimum((v_eff - back_emf) / self._R, torch.full_like(back_emf, self._i_peak))
        i_lo = torch.maximum((-v_eff - back_emf) / self._R, torch.full_like(back_emf, -self._i_peak))
        # if the back-EMF exceeds the bus the motor can only brake (regen): keep the interval sane
        i_hi = torch.maximum(i_hi, torch.zeros_like(i_hi) - 1e-6)
        i_lo = torch.minimum(i_lo, torch.zeros_like(i_lo) + 1e-6)
        i_des = tau_des / self._kt_out
        i = torch.clamp(i_des, min=i_lo, max=i_hi)
        tau_motor = self._kt_out * i

        # ---- friction budget (inside the actuator; PhysX joint friction is 0) -----------------
        sgn = torch.tanh(joint_vel / 0.05)
        stribeck = self.cfg.stribeck_friction * torch.exp(-((joint_vel / self.cfg.stribeck_velocity) ** 2))
        load_dep = 1.0 + self.cfg.load_dep_friction * tau_motor.abs() / self._tau_peak
        tau_fric = (self.cfg.coulomb_friction + stribeck) * sgn * load_dep + self.cfg.viscous_friction * joint_vel
        tau_fric = tau_fric * self.friction_scale
        tau_out = torch.clamp(tau_motor - tau_fric, min=-self._tau_peak, max=self._tau_peak)

        # ---- book-keeping --------------------------------------------------------------------
        self.current = i
        self.motor_torque = tau_motor
        self.friction_torque = tau_fric
        self.saturated = (i.abs() >= 0.98 * self._i_peak) | (tau_out.abs() >= 0.98 * self._tau_peak)
        self.voltage_limited = (i_des > i_hi + 1e-6) | (i_des < i_lo - 1e-6)
        self._sag_prev = tau_out.abs().sum(dim=1, keepdim=True)
        self.applied_effort = tau_out

        control_action.joint_efforts = tau_out
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    # ------------------------------------------------------------------------------------------
    @property
    def continuous_torque(self) -> float:
        return self._tau_cont

    @property
    def peak_torque(self) -> float:
        return self._tau_peak
