"""First-trial evaluation of trained native manager skills; no pose assistance."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument("--skill",choices=["stand","walk","standup","backflip"],required=True)
p.add_argument("--checkpoint",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
p.add_argument("--num_envs",type=int,default=16)
p.add_argument("--steps",type=int,default=500)
p.add_argument("--seed",type=int,default=9127)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from wheel_humanoid_lab.tasks.manager_based.inertia.native_skills import CONTRACT
from skills.policy import FrozenActor

env=None
try:
    task=f"Isaac-Q1-Native-{a.skill.capitalize()}-v0"
    cfg=parse_env_cfg(task,device=a.device or "cuda:0",num_envs=a.num_envs)
    cfg.seed=a.seed;cfg.curriculum=None;cfg.observations.policy.enable_corruption=False
    env=gym.make(task,cfg=cfg).unwrapped
    assert isinstance(env,ManagerBasedRLEnv)
    env.reset(seed=a.seed);s=env.command_manager.get_term("skill");s.level=1.
    actor=FrozenActor(a.checkpoint,env.device)
    assert actor.input_dim==90 and actor.mlp[-1].out_features==23
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=env.device)
    successes=torch.zeros_like(alive);illegal=torch.zeros_like(alive);takeoff=torch.zeros_like(alive)
    best_hold=torch.zeros(a.num_envs,device=env.device);max_turn=best_hold.clone()
    samples=upright_count=velocity_error=yaw_error=failures=nonfinite=0
    trace=[]
    with torch.inference_mode():
        for step in range(a.steps):
            command=env.command_manager.get_term("base_velocity")
            command.command[:]=torch.tensor([.4 if a.skill=="walk" else 0.,0.,0.],device=env.device)
            command.time_left.fill_(1.e9);command.is_standing_env.fill_(a.skill!="walk")
            s._update_command();obs=env.observation_manager.compute()["policy"]
            actions=actor(obs)
            if not torch.isfinite(actions).all():raise FloatingPointError("Evaluation action is non-finite")
            _,_,terminated,truncated,_=env.step(actions)
            active=alive & ~terminated & ~truncated
            nonfinite+=int((env.termination_manager.get_term("nonfinite") & alive).sum())
            failures+=int((terminated & alive).sum())
            r=env.scene["robot"].data;z=r.root_pos_w[:,2]-env.scene.env_origins[:,2]
            successes |= s.success & active
            illegal |= getattr(env,"_native_last_illegal",s.illegal) & alive
            takeoff |= s.has_air & active
            best_hold=torch.maximum(best_hold,torch.where(active,s.best_hold,0.))
            max_turn=torch.maximum(max_turn,torch.where(active,-s.winding,0.))
            samples+=int(active.sum())
            upright_count+=int((active & (-r.projected_gravity_b[:,2]>.9) & (z>.65)).sum())
            velocity_error+=float(((r.root_lin_vel_b[:,0]-(.4 if a.skill=="walk" else 0.)).abs()*active).sum())
            yaw_error+=float((r.root_ang_vel_b[:,2].abs()*active).sum())
            if step%25==0:trace.append(dict(step=step,height=z.tolist(),upright=(-r.projected_gravity_b[:,2]).tolist(),active=active.tolist()))
            alive &= ~terminated & ~truncated
            if not alive.any():break
    upright_fraction=upright_count/max(samples,1);speed_error=velocity_error/max(samples,1);yaw_error/=max(samples,1)
    fraction=float(successes.float().mean());failure_fraction=failures/a.num_envs
    if a.skill=="walk":passed=upright_fraction>=.95 and failure_fraction<=.1 and speed_error<.2 and yaw_error<.25
    elif a.skill=="backflip":passed=fraction>=.7 and int(illegal.sum())==0 and nonfinite==0
    else:passed=fraction>=.8 and failure_fraction<=.1 and nonfinite==0
    report=dict(contract=CONTRACT,skill=a.skill,task=task,manager_based=True,first_trial_only=True,
                checkpoint_sha256=actor.checkpoint_sha256,checkpoint_iteration=actor.training_iteration,
                seed=a.seed,trials=a.num_envs,steps=step+1,successes=int(successes.sum()),takeoffs=int(takeoff.sum()),
                failures=failures,illegal=int(illegal.sum()),nonfinite=nonfinite,
                upright_fraction=upright_fraction,velocity_error_mps=speed_error,yaw_error_rps=yaw_error,
                best_hold_s=best_hold.tolist(),max_backturn_rad=max_turn.tolist(),pass_validation=passed,
                score=fraction+upright_fraction-speed_error-failure_fraction,trace=trace)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print("NATIVE_EVALUATION",json.dumps({k:v for k,v in report.items() if k!="trace"}),flush=True)
finally:
    if env is not None:env.close()
    app.close()
