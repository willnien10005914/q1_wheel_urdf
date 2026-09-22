import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";

const URDF_URL = new URL("../urdf/wheel_humanoid.urdf", import.meta.url).href;
const WHEEL_OFFSET = 0.8867;
const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;

const GROUPS = [
  {
    id: "waist",
    title: "Waist",
    joints: ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "neck_joint"],
  },
  {
    id: "larm",
    title: "Left arm",
    joints: [
      "l_shoulder_pitch_joint",
      "l_shoulder_roll_joint",
      "l_shoulder_yaw_joint",
      "l_elbow_joint",
      "l_wrist_joint",
      "l_gripper_joint",
    ],
  },
  {
    id: "rarm",
    title: "Right arm",
    joints: [
      "r_shoulder_pitch_joint",
      "r_shoulder_roll_joint",
      "r_shoulder_yaw_joint",
      "r_elbow_joint",
      "r_wrist_joint",
      "r_gripper_joint",
    ],
  },
  {
    id: "lleg",
    title: "Left leg",
    joints: ["l_hip_pitch_joint", "l_hip_roll_joint", "l_knee_joint", "l_wheel_joint"],
  },
  {
    id: "rleg",
    title: "Right leg",
    joints: ["r_hip_pitch_joint", "r_hip_roll_joint", "r_knee_joint", "r_wheel_joint"],
  },
  {
    id: "mimic",
    title: "Knee rollers (mimic)",
    joints: ["l_knee_roller_joint", "r_knee_roller_joint"],
  },
];

const LABELS = {
  waist_yaw_joint: "Yaw",
  waist_roll_joint: "Roll",
  waist_pitch_joint: "Pitch",
  l_shoulder_pitch_joint: "Shoulder pitch",
  l_shoulder_roll_joint: "Shoulder roll",
  l_shoulder_yaw_joint: "Shoulder yaw",
  neck_joint: "Neck",
  l_elbow_joint: "Elbow",
  l_wrist_joint: "Wrist",
  l_gripper_joint: "Gripper",
  r_shoulder_pitch_joint: "Shoulder pitch",
  r_shoulder_roll_joint: "Shoulder roll",
  r_shoulder_yaw_joint: "Shoulder yaw",
  r_elbow_joint: "Elbow",
  r_wrist_joint: "Wrist",
  r_gripper_joint: "Gripper",
  l_hip_pitch_joint: "Hip pitch",
  l_hip_roll_joint: "Hip roll",
  l_knee_joint: "Knee",
  l_wheel_joint: "Wheel",
  r_hip_pitch_joint: "Hip pitch",
  r_hip_roll_joint: "Hip roll",
  r_knee_joint: "Knee",
  r_wheel_joint: "Wheel",
  l_knee_roller_joint: "Left knee roller",
  r_knee_roller_joint: "Right knee roller",
};

const MIRROR = {
  l_shoulder_pitch_joint: "r_shoulder_pitch_joint",
  l_shoulder_roll_joint: "r_shoulder_roll_joint",
  l_shoulder_yaw_joint: "r_shoulder_yaw_joint",
  l_elbow_joint: "r_elbow_joint",
  l_hip_pitch_joint: "r_hip_pitch_joint",
  l_hip_roll_joint: "r_hip_roll_joint",
  l_knee_joint: "r_knee_joint",
  l_wheel_joint: "r_wheel_joint",
};

const NEGATE_ON_MIRROR = new Set([
  "l_shoulder_roll_joint",
  "r_shoulder_roll_joint",
  "l_hip_roll_joint",
  "r_hip_roll_joint",
]);

Object.entries(MIRROR).forEach(([left, right]) => {
  MIRROR[right] = left;
});

const PRESETS = {
  zero: {},
  kneel: {
    l_hip_pitch_joint: -1.35,
    r_hip_pitch_joint: -1.35,
    l_knee_joint: 1.55,
    r_knee_joint: 1.55,
    l_elbow_joint: -0.6,
    r_elbow_joint: -0.6,
    l_shoulder_pitch_joint: 0.25,
    r_shoulder_pitch_joint: 0.25,
  },
  tpose: {
    l_shoulder_roll_joint: 1.57,
    r_shoulder_roll_joint: -1.57,
  },
  reach: {
    l_shoulder_pitch_joint: -1.2,
    r_shoulder_pitch_joint: -1.2,
    l_elbow_joint: -0.4,
    r_elbow_joint: -0.4,
    waist_pitch_joint: 0.25,
  },
};

const API =
  new URLSearchParams(location.search).get("api") ||
  (location.port === "8766" ? "" : "http://127.0.0.1:8766");

const els = {
  viewport: document.getElementById("viewport"),
  joints: document.getElementById("joints"),
  json: document.getElementById("joint-json"),
  loader: document.getElementById("loader"),
  bar: document.getElementById("load-bar"),
  status: document.getElementById("load-status"),
  error: document.getElementById("error"),
  mirror: document.getElementById("mirror"),
  spin: document.getElementById("spin-wheels"),
  isaac: document.getElementById("isaac-status"),
  cmd: document.getElementById("cmd-readout"),
  pose: document.getElementById("pose-readout"),
  train: document.getElementById("train-status"),
};

const sliderEls = {};
const valueEls = {};
const extraEls = {};
const followBtns = {};
const jointRows = {};
const overrides = new Set();
const dragging = new Set();
let robot = null;
let applying = false;
let anim = null;
let isaacConnected = false;
let pendingOverrides = {};
let lastIsaac = null;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
els.viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c1016);
scene.fog = new THREE.Fog(0x0c1016, 6, 16);

const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 40);
camera.position.set(1.85, 1.15, 1.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.62, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 0.5;
controls.maxDistance = 8;

scene.add(new THREE.HemisphereLight(0xc9d6ea, 0x2a241c, 1.15));
const key = new THREE.DirectionalLight(0xfff4e8, 1.35);
key.position.set(2.2, 3.4, 1.4);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.2;
key.shadow.camera.far = 12;
key.shadow.camera.left = -2.5;
key.shadow.camera.right = 2.5;
key.shadow.camera.top = 2.5;
key.shadow.camera.bottom = -2.5;
scene.add(key);
scene.add(new THREE.DirectionalLight(0x7aa7c9, 0.35).translateX(-2).translateY(1.4).translateZ(-1.6));

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(3.2, 72),
  new THREE.MeshStandardMaterial({ color: 0x161c26, roughness: 0.92, metalness: 0.05 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(4, 16, 0x3a4a60, 0x222b38);
grid.position.y = 0.001;
scene.add(grid);

const origin = new THREE.AxesHelper(0.18);
origin.position.y = 0.002;
scene.add(origin);

function resize() {
  const w = els.viewport.clientWidth;
  const h = els.viewport.clientHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function fmtDeg(rad) {
  return `${(num(rad) * RAD2DEG).toFixed(1)}°`;
}

function fmtMotor(name, pos, vel, tau) {
  if (name.endsWith("_wheel_joint")) {
    return `${num(vel).toFixed(2)} rad/s · τ ${num(tau).toFixed(2)}`;
  }
  return `${fmtDeg(pos)} · τ ${num(tau).toFixed(2)}`;
}

function isWheel(name) {
  return name.endsWith("_wheel_joint");
}

function setConnected(on) {
  isaacConnected = on;
  if (!els.isaac) return;
  els.isaac.className = `status ${on ? "online" : "offline"}`;
  els.isaac.textContent = on
    ? "Isaac Sim: connected (sliders override motors)"
    : "Isaac Sim: offline";
}

async function postJson(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

function queueOverride(name, value) {
  overrides.add(name);
  pendingOverrides[name] = value;
  if (jointRows[name]) jointRows[name].classList.add("overridden");
  if (followBtns[name]) followBtns[name].hidden = false;
}

async function flushOverrides() {
  if (!isaacConnected || !Object.keys(pendingOverrides).length) return;
  const payload = { overrides: pendingOverrides };
  pendingOverrides = {};
  try {
    await postJson("/api/joints", payload);
  } catch {
    /* Isaac play not running */
  }
}

async function releaseJoint(name) {
  overrides.delete(name);
  delete pendingOverrides[name];
  if (jointRows[name]) jointRows[name].classList.remove("overridden");
  if (followBtns[name]) followBtns[name].hidden = true;
  if (!isaacConnected) return;
  try {
    await postJson("/api/joints", { release: [name] });
  } catch {
    /* ignore */
  }
}

async function releaseAll() {
  overrides.clear();
  pendingOverrides = {};
  Object.values(jointRows).forEach((el) => el.classList.remove("overridden"));
  Object.values(followBtns).forEach((btn) => {
    btn.hidden = true;
  });
  if (!isaacConnected) return;
  try {
    await postJson("/api/release", {});
  } catch {
    /* ignore */
  }
}

function applyIsaacState(state) {
  lastIsaac = state;
  if (els.cmd && state.cmd) {
    els.cmd.textContent = `vx ${num(state.cmd.vx).toFixed(2)} · yaw ${num(state.cmd.yaw).toFixed(2)} · z ${
      Number.isFinite(state.base?.z) ? state.base.z.toFixed(2) : "--"
    } m`;
  }
  if (els.pose) {
    const pol = state.policies || {};
    const bits = [
      pol.posture ? "kneel-ppo" : "kneel-scripted",
      pol.slide ? "slide-ppo" : "slide-pending",
    ];
    els.pose.textContent = `pose ${state.pose || "ppo"} · ${bits.join(" · ")}`;
  }
  if (els.train && state.train) {
    els.train.textContent = fmtTrain(state.train);
  }
  const joints = state.joints || {};
  const liveOverrides = new Set(Object.keys(state.overrides || {}));
  Object.keys(joints).forEach((name) => {
    const j = joints[name];
    if (!robot?.joints[name]) return;
    applying = true;
    setJoint(name, j.pos, { fromSlider: true, mirrored: true, fromIsaac: true });
    applying = false;
    if (extraEls[name]) extraEls[name].textContent = fmtMotor(name, j.pos, j.vel, j.tau);
    const held = overrides.has(name) || liveOverrides.has(name) || dragging.has(name);
    if (!held && sliderEls[name]) {
      sliderEls[name].value = String(isWheel(name) ? j.vel : j.pos);
    }
    if (liveOverrides.has(name)) {
      overrides.add(name);
      if (jointRows[name]) jointRows[name].classList.add("overridden");
      if (followBtns[name]) followBtns[name].hidden = false;
    }
  });
}

async function pollIsaac() {
  try {
    const res = await fetch(`${API}/api/state`, { cache: "no-store" });
    if (!res.ok) throw new Error("offline");
    const state = await res.json();
    if (!state.ok) throw new Error("offline");
    if (!isaacConnected) setConnected(true);
    applyIsaacState(state);
    await flushOverrides();
  } catch {
    if (isaacConnected) setConnected(false);
  }
}

function jointLimit(joint) {
  if (joint.jointType === "continuous") return { lower: -Math.PI, upper: Math.PI };
  const lower = Number.isFinite(joint.limit?.lower) ? joint.limit.lower : -Math.PI;
  const upper = Number.isFinite(joint.limit?.upper) ? joint.limit.upper : Math.PI;
  return { lower, upper };
}

function clamp(name, value) {
  const joint = robot.joints[name];
  const { lower, upper } = jointLimit(joint);
  return Math.min(upper, Math.max(lower, value));
}

function setJoint(name, value, { fromSlider = false, mirrored = false, fromIsaac = false } = {}) {
  if (!robot || !robot.joints[name]) return;
  const next = clamp(name, value);
  robot.setJointValue(name, next);
  if (!fromSlider && sliderEls[name]) sliderEls[name].value = String(next);
  if (valueEls[name]) valueEls[name].textContent = fmtDeg(next);

  const joint = robot.joints[name];
  (joint.mimicJoints || []).forEach((mimic) => {
    const mv = mimic.angle ?? mimic.multiplier * next + mimic.offset;
    if (sliderEls[mimic.name]) sliderEls[mimic.name].value = String(mv);
    if (valueEls[mimic.name]) valueEls[mimic.name].textContent = fmtDeg(mv);
  });

  if (!mirrored && els.mirror.checked && MIRROR[name]) {
    const other = MIRROR[name];
    const otherVal = NEGATE_ON_MIRROR.has(name) ? -next : next;
    setJoint(other, otherVal, { mirrored: true });
  }
}

function snapshot() {
  const out = {};
  if (!robot) return out;
  Object.keys(robot.joints).forEach((name) => {
    const joint = robot.joints[name];
    if (!joint || joint.jointType === "fixed") return;
    out[name] = Number(num(joint.angle).toFixed(4));
  });
  return out;
}

function refreshJson() {
  els.json.value = JSON.stringify(snapshot(), null, 2);
}

function applyPose(values, duration = 0.55) {
  if (!robot) return;
  if (anim) cancelAnimationFrame(anim.raf);
  const names = Object.keys(robot.joints).filter((n) => !robot.joints[n].mimicJoint);
  const start = {};
  const end = {};
  names.forEach((name) => {
    start[name] = robot.joints[name].angle || 0;
    end[name] = values[name] ?? 0;
  });
  const t0 = performance.now();
  const tick = (now) => {
    const u = Math.min(1, (now - t0) / (duration * 1000));
    const s = 1 - (1 - u) ** 3;
    applying = true;
    names.forEach((name) => {
      setJoint(name, start[name] + (end[name] - start[name]) * s, { mirrored: true });
    });
    applying = false;
    refreshJson();
    if (u < 1) anim = { raf: requestAnimationFrame(tick) };
    else {
      anim = null;
      if (isaacConnected) {
        names.forEach((name) => {
          if (robot.joints[name]?.mimicJoint) return;
          queueOverride(name, isWheel(name) ? 0 : end[name] ?? 0);
        });
        flushOverrides();
      }
    }
  };
  anim = { raf: requestAnimationFrame(tick) };
}

function highlight(name) {
  Object.values(jointRows).forEach((el) => el.classList.remove("active"));
  if (name && jointRows[name]) {
    jointRows[name].classList.add("active");
    jointRows[name].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function buildSliders() {
  els.joints.innerHTML = "";
  GROUPS.forEach((group) => {
    const wrap = document.createElement("details");
    wrap.className = "group";
    wrap.open = group.id !== "mimic";
    const summary = document.createElement("summary");
    summary.innerHTML = `<span>${group.title}</span><span class="count">${group.joints.length}</span>`;
    wrap.appendChild(summary);

    group.joints.forEach((name) => {
      const joint = robot.joints[name];
      if (!joint) return;
      const { lower, upper } = jointLimit(joint);
      const mimic = Boolean(joint.mimicJoint);
      const wheel = isWheel(name);
      const row = document.createElement("div");
      row.className = `joint${mimic ? " mimic" : ""}`;
      row.innerHTML = `
        <div class="joint-row">
          <div>
            <div class="joint-name">${LABELS[name] || name}</div>
            <div class="joint-key">${name}${mimic ? " · mimic" : ""}</div>
          </div>
          <div class="joint-meta">
            <div class="joint-val" data-val="${name}">0.0°</div>
            <div class="joint-extra" data-extra="${name}"></div>
            <button type="button" class="follow-btn" hidden>Follow PPO</button>
          </div>
        </div>
        <input type="range" min="${wheel ? -25 : lower}" max="${wheel ? 25 : upper}" step="0.001" value="0" ${mimic ? "disabled" : ""}>
        <div class="limits"><span>${wheel ? "-25 rad/s" : fmtDeg(lower)}</span><span>${wheel ? "+25 rad/s" : fmtDeg(upper)}</span></div>
      `;
      const input = row.querySelector("input");
      sliderEls[name] = input;
      valueEls[name] = row.querySelector("[data-val]");
      extraEls[name] = row.querySelector("[data-extra]");
      followBtns[name] = row.querySelector(".follow-btn");
      jointRows[name] = row;
      input.addEventListener("pointerdown", () => {
        highlight(name);
        dragging.add(name);
      });
      input.addEventListener("pointerup", () => dragging.delete(name));
      input.addEventListener("pointercancel", () => dragging.delete(name));
      input.addEventListener("input", () => {
        if (applying) return;
        const value = Number(input.value);
        if (!wheel) setJoint(name, value, { fromSlider: true });
        else if (valueEls[name]) valueEls[name].textContent = `${value.toFixed(2)} rad/s`;
        refreshJson();
        if (isaacConnected && !mimic) queueOverride(name, value);
      });
      followBtns[name].addEventListener("click", (ev) => {
        ev.preventDefault();
        releaseJoint(name);
      });
      wrap.appendChild(row);
    });
    els.joints.appendChild(wrap);
  });
}

function findLink(obj) {
  while (obj) {
    if (obj.isURDFLink) return obj;
    obj = obj.parent;
  }
  return null;
}

function jointForLink(linkName) {
  return Object.values(robot.joints).find((j) => j.children[0]?.name === linkName);
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener("pointerdown", (ev) => {
  if (!robot || ev.button !== 0) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(robot, true);
  if (!hits.length) return;
  const link = findLink(hits[0].object);
  if (!link) return;
  const joint = jointForLink(link.name);
  if (joint) highlight(joint.name);
});

function setView(kind) {
  const looks = {
    front: [[1.9, 0.9, 0], [0, 0.62, 0]],
    side: [[0, 0.9, 2.1], [0, 0.62, 0]],
    top: [[0.05, 2.6, 0.05], [0, 0.2, 0]],
    iso: [[1.85, 1.15, 1.55], [0, 0.62, 0]],
  };
  const [pos, tgt] = looks[kind];
  camera.position.set(...pos);
  controls.target.set(...tgt);
}

document.querySelectorAll("[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setView(btn.dataset.view));
});
document.getElementById("reset").addEventListener("click", () => applyPose(PRESETS.zero));
document.getElementById("follow-all").addEventListener("click", () => releaseAll());
document.getElementById("reset-sim").addEventListener("click", async () => {
  await releaseAll();
  try {
    await postJson("/api/reset", {});
  } catch {
    /* ignore */
  }
});

async function requestPose(name) {
  await releaseAll();
  try {
    await postJson("/api/pose", { name });
  } catch {
    /* ignore */
  }
}

document.getElementById("pose-kneel").addEventListener("click", () => requestPose("kneel"));
document.getElementById("pose-stand").addEventListener("click", () => requestPose("stand"));
document.getElementById("pose-slide").addEventListener("click", () => requestPose("slide"));
document.getElementById("pose-skate").addEventListener("click", () => requestPose("skate"));

function fmtTrain(payload) {
  const jobs = payload.jobs || payload;
  const parts = [];
  ["posture", "slide"].forEach((name) => {
    const j = jobs[name];
    if (!j) return;
    if (j.running) parts.push(`${name} training`);
    else if (j.checkpoint) parts.push(`${name} ready`);
    else parts.push(`${name} none`);
  });
  return `train: ${parts.join(" · ") || "idle"}`;
}

async function requestTrain(name) {
  try {
    const res = await postJson("/api/train", { name });
    if (els.train) {
      els.train.textContent = res.ok
        ? `train: started ${res.label || name} (pid ${res.pid})`
        : `train: ${res.error || "failed"}`;
    }
    if (!res.ok) window.alert(res.error || "Could not start training");
  } catch (err) {
    if (els.train) els.train.textContent = "train: API offline";
    window.alert("Train API is not running. Use ./train_slide.sh or ./train_posture.sh");
  }
}

document.getElementById("train-posture").addEventListener("click", () => requestTrain("posture"));
document.getElementById("train-slide").addEventListener("click", () => requestTrain("slide"));

async function pollTrain() {
  try {
    const res = await fetch(`${API}/api/train`, { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();
    if (els.train) els.train.textContent = fmtTrain(payload);
  } catch {
    /* ignore */
  }
}
setInterval(pollTrain, 4000);
pollTrain();
document.getElementById("copy").addEventListener("click", async () => {
  refreshJson();
  await navigator.clipboard.writeText(els.json.value);
  document.getElementById("copy").textContent = "Copied";
  setTimeout(() => {
    document.getElementById("copy").textContent = "Copy JSON";
  }, 900);
});
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => applyPose(PRESETS[btn.dataset.preset] || {}));
});

function loadRobot() {
  if (location.protocol === "file:") {
    els.status.textContent = "Open this page from a local server, not as a file:// URL.";
    els.error.style.display = "block";
    els.error.textContent = "Run python web/serve.py then open http://127.0.0.1:8765/web/";
    return;
  }

  const manager = new THREE.LoadingManager();
  manager.onProgress = (_url, loaded, total) => {
    const pct = total ? Math.round((loaded / total) * 100) : 0;
    els.bar.style.width = `${pct}%`;
    els.status.textContent = `Loading meshes ${loaded} / ${total}`;
  };

  const loader = new URDFLoader(manager);
  loader.parseCollision = false;
  loader.load(
    URDF_URL,
    (model) => {
      robot = model;
      robot.rotation.x = -Math.PI / 2;
      robot.position.y = WHEEL_OFFSET;
      robot.traverse((obj) => {
        if (obj.isMesh) {
          obj.castShadow = true;
          obj.receiveShadow = true;
          if (obj.material) {
            obj.material.side = THREE.DoubleSide;
            obj.material.shininess = 38;
          }
        }
      });
      scene.add(robot);
      try {
        buildSliders();
        refreshJson();
      } catch (err) {
        els.error.style.display = "block";
        els.error.textContent = String(err);
      }
      els.loader.classList.add("hidden");
    },
    null,
    (err) => {
      els.error.style.display = "block";
      els.error.textContent = String(err);
      els.status.textContent = "Load failed";
    }
  );
}

let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (robot && els.spin.checked && !isaacConnected) {
    ["l_wheel_joint", "r_wheel_joint"].forEach((name) => {
      const cur = robot.joints[name].angle || 0;
      let next = cur + dt * 2.4;
      if (next > Math.PI) next -= Math.PI * 2;
      setJoint(name, next, { mirrored: true });
    });
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

window.addEventListener("resize", resize);
resize();
loadRobot();
setInterval(pollIsaac, 50);
requestAnimationFrame(loop);
