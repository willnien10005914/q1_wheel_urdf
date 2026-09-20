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
    title: "腰部",
    joints: ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
  },
  {
    id: "larm",
    title: "左臂",
    joints: [
      "l_shoulder_pitch_joint",
      "l_shoulder_roll_joint",
      "l_shoulder_yaw_joint",
      "l_elbow_joint",
    ],
  },
  {
    id: "rarm",
    title: "右臂",
    joints: [
      "r_shoulder_pitch_joint",
      "r_shoulder_roll_joint",
      "r_shoulder_yaw_joint",
      "r_elbow_joint",
    ],
  },
  {
    id: "lleg",
    title: "左腿",
    joints: ["l_hip_pitch_joint", "l_hip_roll_joint", "l_knee_joint", "l_wheel_joint"],
  },
  {
    id: "rleg",
    title: "右腿",
    joints: ["r_hip_pitch_joint", "r_hip_roll_joint", "r_knee_joint", "r_wheel_joint"],
  },
  {
    id: "mimic",
    title: "膝蓋連動滾輪",
    joints: ["l_knee_roller_joint", "r_knee_roller_joint"],
  },
];

const LABELS = {
  waist_yaw_joint: "Yaw 偏航",
  waist_roll_joint: "Roll 側傾",
  waist_pitch_joint: "Pitch 俯仰",
  l_shoulder_pitch_joint: "肩 Pitch",
  l_shoulder_roll_joint: "肩 Roll",
  l_shoulder_yaw_joint: "肩 Yaw",
  l_elbow_joint: "肘",
  r_shoulder_pitch_joint: "肩 Pitch",
  r_shoulder_roll_joint: "肩 Roll",
  r_shoulder_yaw_joint: "肩 Yaw",
  r_elbow_joint: "肘",
  l_hip_pitch_joint: "髖 Pitch",
  l_hip_roll_joint: "髖 Roll",
  l_knee_joint: "膝",
  l_wheel_joint: "輪",
  r_hip_pitch_joint: "髖 Pitch",
  r_hip_roll_joint: "髖 Roll",
  r_knee_joint: "膝",
  r_wheel_joint: "輪",
  l_knee_roller_joint: "左膝滾輪",
  r_knee_roller_joint: "右膝滾輪",
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
};

const sliderEls = {};
const valueEls = {};
const jointRows = {};
let robot = null;
let applying = false;
let anim = null;

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

function fmtDeg(rad) {
  return `${(rad * RAD2DEG).toFixed(1)}°`;
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

function setJoint(name, value, { fromSlider = false, mirrored = false } = {}) {
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
    out[name] = Number(robot.joints[name].angle.toFixed(4));
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
    else anim = null;
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
      const row = document.createElement("div");
      row.className = `joint${mimic ? " mimic" : ""}`;
      row.innerHTML = `
        <div class="joint-row">
          <div>
            <div class="joint-name">${LABELS[name] || name}</div>
            <div class="joint-key">${name}${mimic ? " · mimic" : ""}</div>
          </div>
          <div class="joint-val" data-val="${name}">0.0°</div>
        </div>
        <input type="range" min="${lower}" max="${upper}" step="0.001" value="0" ${mimic ? "disabled" : ""}>
        <div class="limits"><span>${fmtDeg(lower)}</span><span>${fmtDeg(upper)}</span></div>
      `;
      const input = row.querySelector("input");
      sliderEls[name] = input;
      valueEls[name] = row.querySelector("[data-val]");
      jointRows[name] = row;
      input.addEventListener("input", () => {
        if (applying) return;
        setJoint(name, Number(input.value), { fromSlider: true });
        refreshJson();
      });
      input.addEventListener("pointerdown", () => highlight(name));
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
document.getElementById("copy").addEventListener("click", async () => {
  refreshJson();
  await navigator.clipboard.writeText(els.json.value);
  document.getElementById("copy").textContent = "已複製";
  setTimeout(() => {
    document.getElementById("copy").textContent = "複製 JSON";
  }, 900);
});
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => applyPose(PRESETS[btn.dataset.preset] || {}));
});

function loadRobot() {
  if (location.protocol === "file:") {
    els.status.textContent = "請用本機伺服器開啟，不要直接雙擊 HTML。";
    els.error.style.display = "block";
    els.error.textContent = "執行 python web/serve.py 後再開 http://127.0.0.1:8765/web/";
    return;
  }

  const manager = new THREE.LoadingManager();
  manager.onProgress = (_url, loaded, total) => {
    const pct = total ? Math.round((loaded / total) * 100) : 0;
    els.bar.style.width = `${pct}%`;
    els.status.textContent = `載入網格 ${loaded} / ${total}`;
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
      buildSliders();
      refreshJson();
      els.loader.classList.add("hidden");
    },
    null,
    (err) => {
      els.error.style.display = "block";
      els.error.textContent = String(err);
      els.status.textContent = "載入失敗";
    }
  );
}

let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (robot && els.spin.checked) {
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
requestAnimationFrame(loop);
