# Q1 Wheel / BI2 Wheel — CubeMars actuator sheet

Source slide: `docs/spec/bi2_wheel_motor_overview.png` (Quanta BU10, "Project Overview — BI2 Wheel").
Machine-readable copy: `config/q1_wheel_components.yaml` (consumed by `tools/build_urdf.py` and the
Isaac Lab actuator groups in `wheel_humanoid_lab/assets/wheel_humanoid.py`).

## Joint → motor map (24 actuated DoF)

| Group    | Count | Joints (URDF)                                            | Motor                     | Slide τ_peak | D (mm) |
|----------|-------|----------------------------------------------------------|---------------------------|--------------|--------|
| Neck     | 1     | `neck_joint`                                             | AK45-36 V3.0 KV80         | 24 Nm        | 55     |
| Gripper  | 1×2   | `l/r_gripper_joint`                                      | AK45-10 (V3.0 KV75)       | 7 Nm         | 53     |
| Shoulder | 3×2   | `l/r_shoulder_{pitch,roll,yaw}_joint`                    | AK10-9 V3.0 KV60          | 43 Nm        | 98     |
| Elbow    | 1×2   | `l/r_elbow_joint`                                        | AK10-9 V3.0 KV60          | 43 Nm        | 98     |
| Wrist    | 1×2   | `l/r_wrist_joint`                                        | AK45-10 (V3.0 KV75)       | 7 Nm         | 53     |
| Torso    | 3     | `waist_{yaw,roll,pitch}_joint`                           | AK10-9 V3.0 KV60          | 43 Nm        | 98     |
| Hip      | 2×2   | `l/r_hip_{pitch,roll}_joint`                             | AKE90-8 KV35              | 121 Nm       | 107.5  |
| Knee     | 1×2   | `l/r_knee_joint`                                         | AKE90-8 KV35              | 121 Nm       | 107.5  |
| Wheel    | 1×2   | `l/r_wheel_joint` (**active** drive, velocity command)   | AK10-9 V3.0 KV60          | 43 Nm        | 98     |

Neck, wrist and gripper have no mesh in the 2026-09-11 STL set; `tools/build_urdf.py` adds them as real
revolute joints carrying the motor housing only (the head shell is fused into the torso mesh). They are
frozen at 0 for the skate task but stay in the 24-D action/observation contract.

## CubeMars datasheet numbers (cubemars.com, Sep 2026)

| SKU                | V_bus | KV (rpm/V) | Kt (Nm/A, motor) | R φ-φ (mΩ) | L (µH) | Gear | I_rated / I_peak (A) | τ_rated / τ_peak (Nm) | ω_no-load (rpm) | J_rotor (g·cm²) | mass (g) |
|--------------------|-------|-----------:|-----------------:|-----------:|-------:|-----:|---------------------:|----------------------:|----------------:|----------------:|---------:|
| AKE90-8 KV35       | 48    | 35         | 0.272            | 164        | 235    | 8:1  | 21 / 72              | 55 / **170**          | 210             | 3377            | 1400     |
| AK10-9 V3.0 KV60   | 48    | 60         | 0.160            | 248        | 213    | 9:1  | 10.7 / 31.9          | 18 / **53**           | 320             | 1002            | 940      |
| AK45-36 V3.0 KV80  | 24    | 80         | 0.110            | 1800       | 1100   | 36:1 | 2.0 / 6.5            | 8 / 24                | 52              | 182             | 349      |
| AK45-10 V3.0 KV75  | 24    | 75         | 0.127            | 2200       | 1330   | 10:1 | 1.9 / 5.0            | 2.5 / 7               | 180             | 157             | 262      |

Sources: [AKE90-8 KV35](https://www.cubemars.com/product/ake90-8-kv35-quasi-direct-drive-actuator.html),
[AK10-9 V3.0 KV60](https://www.cubemars.com/product/ak10-9-v3-0-kv60-robotic-actuator.html),
[AK45-36 V3.0 KV80](https://www.cubemars.com/product/ak45-36-v3-0-kv80-robotic-actuator.html),
[AK45-10 V3.0 KV75](https://www.cubemars.com/product/ak45-10-v3-0-kv75-robotic-actuator.html).

Notes / assumptions (also in the YAML comments):

* The slide prints **121 Nm** for AKE90-8 and **43 Nm** for AK10-9; the current datasheets say 170 Nm
  (121 Nm/kg is the torque density) and 53 Nm (V2.0 was 48). We use the *slide* values as the operating
  peak torque (conservative effort clip) and keep the datasheet peak for reference.
* Continuous torque comes from the sheets (55 / 18 / 8 / 2.5 Nm ≈ 0.33–0.45 τ_peak).
* Kt ≈ 60 / (2π·KV) checks out within 5 % for all four SKUs; the back-EMF constant used in the sim is
  60 / (2π·KV) so the no-load speed matches the sheet at V_bus.
* 48 V pack for AK10-9 / AKE90 (DR 44–50 V), 24 V rail for the AK45 family (DR 22.5–25 V).
* Reflected rotor inertia J·G² is written to PhysX as joint armature: AKE90 0.0216, AK10-9 0.0081,
  AK45-36 0.0236, AK45-10 0.0016 kg·m² (DR ×0.9–1.1).

## Mass budget (40 kg target)

`tools/build_urdf.py --check`:

```
structural (uniform density) :   18.392 kg  -> x0.9318
CubeMars stators (24)        :   22.017 kg
slide-only links (5)         :    0.790 kg
IMU                          :    0.055 kg
target total                 :   40.000 kg
```

Stators are added to the **parent** link at the joint origin (parallel-axis theorem); the CAD
uniform-density estimates are scaled to close the budget. Replace the estimates with CAD masses in
`urdf/wheel_humanoid_structural.urdf` when available and re-run the tool.

## Other components

| Item                | Value                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------|
| Height              | ~1.42 m spec; zero-pose mesh top is 1.37 m above the wheel contact                        |
| Wheel               | r = 0.100 m (200 mm diameter, CAD), width 60 mm, rubber µ 0.8–1.2 (DR 0.6–1.3)            |
| IMU                 | Xsens MTi-630 AHRS on the pelvis at hip-pitch axis height (`imu_link`, 0, 0, −0.1267 m)  |
| IMU specs           | gyro ±2000 °/s, 0.007 °/s/√Hz, 8 °/h; accel ±10 g, 60 µg/√Hz; 400 Hz SDI output           |
| Motor bus           | CAN, 200 Hz command loop → PhysX dt = 5 ms, policy 50 Hz (decimation 4)                  |

## Sim actuator model (`wheel_humanoid_lab.actuators.CubeMarsActuator`)

```
tau_des   = kp (q_d - q) + kd (qd_d - qd) + tau_ff        # position groups
          = kd (qd_d - qd) + tau_ff                        # wheel group (velocity mode)
V_eff     = clip(V_bus - k_sag * sum|tau_applied|, V_min, V_bus)
e         = Ke * G * qd
i         = clip(tau_des / (Kt G eta), [(-V_eff - e)/R, (V_eff - e)/R] ∩ [-I_peak, I_peak])
tau_motor = Kt G eta i
tau_fric  = [tau_c + tau_s exp(-(qd/v_s)^2)] tanh(qd/eps) (1 + k_load |tau_motor|/tau_peak) + b qd
tau_out   = clip(tau_motor - s_fric tau_fric, -tau_peak, tau_peak)
```

Command delay 0–3 physics steps (0–15 ms), bus voltage, and friction scale (0.85–1.15) are per-env and
re-sampled from the nominal range on every reset (restore-then-sample; nothing accumulates). PhysX
joint friction is forced to 0 for these joints so the budget is not counted twice.

Joint-side PD gains (scaled with τ_peak, not copied from XL330-class robots):

| Group          | kp (Nm/rad) | kd (Nm·s/rad) |
|----------------|------------:|--------------:|
| AKE90 hip/knee | 180         | 7.0           |
| AK10 torso     | 70          | 3.0           |
| AK10 arms      | 45          | 2.0           |
| AK10 wheels    | –           | 1.2 (velocity loop) |
| AK45 neck      | 15          | 0.8           |
| AK45 wrist/grip| 8           | 0.4           |
