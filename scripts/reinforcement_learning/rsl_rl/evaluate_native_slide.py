"""First-trial skating validation requires both feet to push and recover."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--skill',choices=['slide'],default='slide')
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=64)
p.add_argument('--steps',type=int,default=1000)
p.add_argument('--seed',type=int,default=9127)
p.add_argument('--out',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from skills.policy import FrozenActor
from wheel_humanoid_lab.tasks.manager_based.native_slide.gait import CONTRACT
env=None
try:
    cfg=parse_env_cfg('Isaac-Q1-Native-Slide-v0',device=a.device or 'cuda:0',num_envs=a.num_envs)
    cfg.curriculum=None;cfg.seed=a.seed;cfg.observations.policy.enable_corruption=False
    # A complete 20-second first trial; no timeout resets during the evaluation.
    cfg.episode_length_s=max(22.,a.steps*.02+1.)
    env=gym.make('Isaac-Q1-Native-Slide-v0',cfg=cfg).unwrapped
    env.reset(seed=a.seed);s=env.command_manager.get_term('skill');s.level=1.
    actor=FrozenActor(a.checkpoint,env.device)
    assert actor.input_dim==90 and actor.mlp[-1].out_features==23
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=env.device)
    failed=torch.zeros_like(alive);cycles=torch.zeros(a.num_envs,2,device=env.device)
    count=upright=push=recovery=hop=error=yaw_error=nonfinite=0.;trace=[]
    with torch.inference_mode():
        for step in range(a.steps):
            vx=0. if step<100 or step>=800 else .4
            yaw=.1 if 600<=step<700 else -.1 if 700<=step<800 else 0.
            cmd=env.command_manager.get_term('base_velocity');cmd.command[:]=torch.tensor([vx,0.,yaw],device=env.device)
            cmd.time_left.fill_(1.e9);cmd.is_standing_env.fill_(not bool(vx or yaw));s._update_command()
            obs=env.observation_manager.compute()['policy'];action=actor(obs)
            assert torch.isfinite(action).all()
            _,_,term,trunc,_=env.step(action)
            failed|=alive&term;nonfinite+=int((env.termination_manager.get_term('nonfinite')&alive).sum())
            active=alive&~term&~trunc;last=env._slide_last
            cycles=torch.maximum(cycles,torch.where(active[:,None],last['cycles'],0.))
            r=env.scene['robot'].data;z=r.root_pos_w[:,2]-env.scene.env_origins[:,2]
            count+=int(active.sum());upright+=int((active&(-r.projected_gravity_b[:,2]>.9)&(z>.65)).sum())
            push+=float((last['push']*active).sum());recovery+=float((last['recovery']*active).sum())
            hop+=float((last['hop']*active).sum());error+=float(((r.root_lin_vel_b[:,0]-vx).abs()*active).sum())
            yaw_error+=float(((r.root_ang_vel_b[:,2]-yaw).abs()*active).sum())
            if step%50==0:trace.append(dict(step=step,active=active.tolist(),height=z.tolist(),cycles=cycles.tolist()))
            alive=active
            if not alive.any():break
    # Standing, wheel-only driving or repeated hopping cannot pass this gate.
    still=(r.root_lin_vel_b.norm(dim=-1)<.2)&(r.root_ang_vel_b.norm(dim=-1)<.5)
    gait_success=(cycles.min(-1).values>=3)&~failed&alive&still
    fraction=float(gait_success.float().mean());up=upright/max(count,1);speed=error/max(count,1)
    yaw_error/=max(count,1)
    push/=max(count,1);recovery/=max(count,1);hop/=max(count,1)
    passed=fraction>=.8 and float(failed.float().mean())<=.1 and up>=.95 and speed<.2 and hop<.1 and yaw_error<.2 and nonfinite==0
    report=dict(contract=CONTRACT,skill='slide',first_trial_only=True,seed=a.seed,trials=a.num_envs,
                checkpoint_sha256=actor.checkpoint_sha256,checkpoint_iteration=actor.training_iteration,
                steps=step+1,valid_cycles_each_foot=cycles.tolist(),successes=int(gait_success.sum()),
                failures=int(failed.sum()),nonfinite=nonfinite,upright_fraction=up,velocity_error_mps=speed,
                yaw_error_rps=yaw_error,stop_stable=still.tolist(),mean_ground_push_reward=push,mean_airborne_forward_reward=recovery,mean_hop_penalty=hop,
                pass_validation=passed,score=fraction+up-speed-hop-float(failed.float().mean()),trace=trace)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('SLIDE_EVALUATION',json.dumps({k:v for k,v in report.items() if k!='trace'}),flush=True)
finally:
    if env is not None:env.close()
    app.close()
