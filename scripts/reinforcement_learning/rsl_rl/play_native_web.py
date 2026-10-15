"""Native CAD robot Web player using the same ManagerBasedRLEnv as training."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--web-port", type=int, default=8875)
p.add_argument("--no-browser", action="store_true")
p.add_argument("--no-keyboard", action="store_true")
p.add_argument("--real-time", action="store_true")
p.add_argument("--steps", type=int, default=0)
p.add_argument("--out", type=Path)
p.add_argument("--sequence", default="")
p.add_argument("--models", type=Path, default=ROOT/"checkpoints/native_manager")
p.add_argument("--training-state", type=Path, default=ROOT/"outputs/native_manager/queue/status.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.steps < 0: p.error("--steps must be nonnegative")
a.models=a.models.expanduser().resolve()
a.training_state=a.training_state.expanduser().resolve()
app = AppLauncher(a).app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from wheel_humanoid_lab.tasks.manager_based.inertia.native_skills import CONTRACT
from skills.policy import FrozenActor
from skills.teleop import IsaacKeyboard
from play_web import start_play_web


def read_json(path, fallback):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return fallback


class NativeWeb:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.skill = env.command_manager.get_term("skill")
        self.motors = env.action_manager.get_term("motors")
        self.mode = "stand"
        self.message = "新版 Q1：平衡控制；等待原生 PPO 模型。"
        self.actors = {}; self.models = {}; self.last_refresh = 0.
        self.command = torch.zeros(1, 3, device=env.device)
        self.training = {}; self.history = []
        self.failed = False

    def refresh(self):
        if time.monotonic()-self.last_refresh < 2.: return
        self.last_refresh = time.monotonic()
        self.training = read_json(a.training_state, {})
        for name in ("stand", "walk", "standup", "backflip"):
            folder = a.models/name
            path = folder/"model.pt"
            validated = path.is_file()
            if not validated: path = folder/"latest.pt"
            if not path.is_file(): continue
            fingerprint = (str(path), path.stat().st_mtime_ns)
            if self.models.get(name, {}).get("fingerprint") == fingerprint: continue
            try:
                actor = FrozenActor(path, self.env.device)
                if actor.input_dim != 90 or actor.mlp[-1].out_features != 23:
                    raise ValueError("Checkpoint does not match native 23-action/90-observation contract")
                self.actors[name] = actor
                self.models[name] = dict(path=str(path.relative_to(ROOT)), sha256=actor.checkpoint_sha256,
                                         iteration=actor.training_iteration, validated=validated,
                                         fingerprint=fingerprint)
                print("NATIVE_WEB_MODEL", name, json.dumps(self.models[name]), flush=True)
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                print("NATIVE_WEB_MODEL_WAIT", name, str(error), flush=True)

    def request(self, name):
        mapping = {"skate":"walk", "walk":"walk", "standard":"stand", "stand":"stand",
                   "unbox":"standup", "getup":"standup", "standup":"standup", "kneel":"kneel", "backflip":"backflip"}
        if name == "reset":
            self.env.reset(); self.mode="stand"; self.failed=False; self.command.zero_()
            self.message="新版 Q1 已重設。"; return True
        if name not in mapping:
            self.message="此動作尚未加入新版原生訓練。"; return False
        target = mapping[name]
        if target == "backflip" and target not in self.actors:
            self.message="後空翻模型仍在佇列中，尚無原生 checkpoint。"; return False
        # A command handover changes manager buffers only.
        before_root=self.robot.data.root_state_w.clone(); before_q=self.robot.data.joint_pos.clone()
        self.skill.set_mode(target); self.skill.level=1.
        assert torch.equal(before_root,self.robot.data.root_state_w) and torch.equal(before_q,self.robot.data.joint_pos)
        self.mode=target; self.failed=False; self.motors.manual={}
        if target != "walk": self.command.zero_()
        trained=self.models.get("standup" if target=="kneel" else target)
        detail="已驗證 PPO" if trained and trained["validated"] else "訓練中的 PPO" if trained else "姿態／平衡控制，PPO 尚待訓練"
        self.message=f"{target} · 新版 URDF · {detail}"
        return True

    def drive(self, values):
        v=[float(values.get(n,0.)) for n in ("vx","vy","yaw")]
        if not all(math.isfinite(x) for x in v): return False
        if self.mode not in ("walk","stand"): return False
        if any(abs(x) > 1.e-6 for x in v) and self.mode != "walk": self.request("walk")
        self.command[0]=torch.tensor([max(-.4,min(1.2,v[0])),0.,max(-.5,min(.5,v[2]))],device=self.env.device)
        return True

    def action(self):
        self.refresh()
        term=self.env.command_manager.get_term("base_velocity")
        term.command[:]=self.command
        term.time_left.fill_(1.e9); term.is_standing_env.fill_(not bool(self.command.abs().max() > 0.))
        self.skill._update_command()
        obs=self.env.observation_manager.compute()["policy"]
        key="standup" if self.mode=="kneel" else self.mode
        actor=self.actors.get(key)
        if actor is None: return torch.zeros(1,23,device=self.env.device)
        return actor(obs)

    def snapshot(self):
        s=self.skill; s.update_state(); r=self.robot.data
        pose={"walk":"skate","standup":"unbox"}.get(self.mode,self.mode)
        if self.mode=="standup" and bool(s.success[0]): pose="unbox_hold"
        if bool(s.illegal[0]): self.failed=True
        if self.failed: self.message="動作未通過接觸驗證；可按 Reset sim 重試。"
        model=self.models.get("standup" if self.mode=="kneel" else self.mode)
        mode_label="已驗證 PPO" if model and model["validated"] else "訓練中 PPO" if model else "平衡／姿態控制"
        if not self.failed: self.message=f"{self.mode} · {mode_label}"+ (" · 已站穩" if pose=="unbox_hold" else "")
        q=r.joint_pos[0].tolist(); qd=r.joint_vel[0].tolist(); tau=r.applied_torque[0].tolist()
        local=r.root_pos_w[0]-self.env.scene.env_origins[0]; quat=r.root_quat_w[0].tolist()
        joints={n:dict(pos=q[i],vel=qd[i],torque=tau[i]) for i,n in enumerate(self.robot.joint_names)}
        policies={n:n in self.actors for n in ("stand","walk","standup","backflip")}
        policies["unbox"]=policies["standup"]
        return dict(ok=True,contract=CONTRACT,environment="ManagerBasedRLEnv",urdf="urdf/q1_inertia/quanta_robot_wheeled.urdf",
                    action_dim=23,observation_dim=90,mass_kg=float(r.default_mass[0].sum()),pose=pose,message=self.message,
                    cmd=dict(vx=float(self.command[0,0]),vy=0.,yaw=float(self.command[0,2])),
                    base=dict(x=float(local[0]),y=float(local[1]),z=float(local[2]),qw=quat[0],qx=quat[1],qy=quat[2],qz=quat[3]),
                    joints=joints,policies=policies,models=self.models,
                    supported_actions=dict(stand=True,skate=True,walk=True,kneel=True,unbox=True,backflip=policies["backflip"]),
                    manual_control_available=self.mode in ("stand","walk"),
                    backflip=dict(success=bool(s.success[0]) and self.mode=="backflip",hold_s=float(s.best_hold[0]),
                                  winding_rad=float(s.winding[0]),airtime_s=float(s.max_air_time[0]),illegal=bool(s.illegal[0])),
                    training=self.training)


env=None; keyboard=None
try:
    cfg=parse_env_cfg("Isaac-Q1-Native-Web-v0",device=a.device or "cuda:0",num_envs=1);cfg.seed=42
    env=gym.make("Isaac-Q1-Native-Web-v0",cfg=cfg).unwrapped
    if not isinstance(env,ManagerBasedRLEnv): raise TypeError("Web requires ManagerBasedRLEnv")
    env.reset(seed=42); runtime=NativeWeb(env)
    state=start_play_web(port=a.web_port,open_browser=not a.no_browser and not a.headless)
    if not a.headless and not a.no_keyboard: keyboard=IsaacKeyboard()
    schedule={int(k):v for k,v in (item.split(":",1) for item in a.sequence.split(",") if item)}
    step=0; requests=[]; records=[]
    with torch.inference_mode():
        while app.is_running() and (not a.steps or step<a.steps):
            started=time.monotonic()
            if state.consume_reset(): runtime.request("reset")
            name=state.consume_pose() or (keyboard.pop_pose(runtime.mode) if keyboard else None) or schedule.get(step)
            if name:
                state.apply_joint_msg({"release_all":True})
                accepted=runtime.request(name); row=dict(step=step,name=name,accepted=accepted)
                requests.append(row); print("NATIVE_WEB_REQUEST",json.dumps(row),flush=True)
            with state._lock: command=state.web_cmd;state.web_cmd=None
            if command is not None: runtime.drive(command)
            if keyboard:
                command=keyboard.command_update()
                if command is not None: runtime.drive(command)
            overrides=state.pop_overrides()
            runtime.motors.manual={k:float(v) for k,v in overrides.items()
                                   if k in runtime.robot.joint_names and math.isfinite(float(v))} if runtime.mode in ("stand","walk") else {}
            _,_,_,_,_=env.step(runtime.action())
            snapshot=runtime.snapshot();state.publish(snapshot)
            if a.out and step%25==0: records.append(dict(step=step,pose=snapshot["pose"],base=snapshot["base"],backflip=snapshot["backflip"]))
            step+=1
            if a.real_time: time.sleep(max(0.,env.step_dt-(time.monotonic()-started)))
    if a.steps and step!=a.steps: raise RuntimeError("Player exited before requested finite steps")
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(dict(steps=step,requests=requests,records=records,final=runtime.snapshot()),indent=2))
finally:
    if keyboard: keyboard.close()
    if env is not None: env.close()
    app.close()
