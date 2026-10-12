"""Deterministic first-episode evaluation and honest final-checkpoint recording."""
import argparse,hashlib,json,math
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--teacher',default='q1_backflip_v2/teacher_balanced.json');p.add_argument('--out',required=True);p.add_argument('--video',action='store_true');p.add_argument('--steps',type=int,default=750)
p.add_argument('--video_num_envs',type=int,default=1);p.add_argument('--video_env_index',type=int,default=0);p.add_argument('--seed',type=int,default=42)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=a.video;app=AppLauncher(a).app
import torch,yaml,numpy as np,importlib
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
ck=Path(a.checkpoint).resolve() if a.checkpoint else None;out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
source=Path('q1_backflip_v2/env.py')
teacher=json.loads((ck.parent/'teacher.json' if ck else Path(a.teacher)).read_text())
environment=teacher.get('environment','env')
if environment not in ['env','compact_env','landing_env']:raise ValueError('Unsupported environment')
module=importlib.import_module(environment);Config,Env=module.Config,module.Env
if ck:
    manifest=json.loads((ck.parent/'manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files']['q1_backflip_v2/env.py'],'Environment differs from training snapshot'
    assert hashlib.sha256(Path('q1_backflip/env.py').read_bytes()).hexdigest()==manifest['files']['q1_backflip/env.py'],'Base environment differs from training snapshot'
    if environment!='env':
        dependency=Path('q1_backflip_v2')/(environment+'.py')
        assert hashlib.sha256(dependency.read_bytes()).hexdigest()==manifest['files'][str(dependency)],'Tuck environment differs from training snapshot'
    if environment=='landing_env':
        dependency=Path('q1_backflip_v2/compact_env.py')
        assert hashlib.sha256(dependency.read_bytes()).hexdigest()==manifest['files'][str(dependency)],'Underlying tuck environment differs from training snapshot'
cfg=Config();cfg.scene.num_envs=a.video_num_envs if a.video else 32;cfg.seed=42;cfg.scene.env_spacing=40. if a.video and a.video_num_envs==1 else 4.
watch=a.video_env_index if a.video else 0
if not 0<=watch<cfg.scene.num_envs:raise ValueError('Invalid video environment index')
cfg.sim.device=a.device or 'cuda:0';cfg.viewer.origin_type='env';cfg.viewer.env_index=watch;cfg.viewer.eye=(3.,-4.,2.6);cfg.viewer.lookat=(0,0,1.1);cfg.viewer.resolution=(960,720)
env=Env(cfg,render_mode='rgb_array' if a.video else None);env.updates=20000
env.set_parameters(torch.tensor(teacher['parameters'],device=env.device).expand(env.num_envs,-1))
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
if ck:
    agent=yaml.safe_load((ck.parent/'agent.yaml').read_text());runner=OnPolicyRunner(wrapped,agent,log_dir=None,device=cfg.sim.device);runner.load(str(ck));policy=runner.get_inference_policy()
else:
    policy=lambda obs:torch.zeros(env.num_envs,24,device=env.device)
results=[];first_seen=set();current_seed=42;original_reset=env._reset_idx
# Capture terminal telemetry BEFORE reset clears evidence.
def capture_reset(ids):
    if ids is None:ids=env.robot._ALL_INDICES
    for i in ids.tolist():
        if (a.video and i==watch) or (not a.video and i not in first_seen):
            results.append({'seed':current_seed,'env':i,'success':bool(env.success[i]),'real_takeoff':bool(env.jumped[i]),'backward_rotation_rad':float(-env.winding[i]),'peak_pelvis_height_m':float(env.peak[i]),'illegal_contact':bool(env.failed[i]),'landing_hold_s':float(env.hold[i]),'duration_s':float(env.episode_length_buf[i])*env.step_dt,'early_hop':bool(env.prehop[i]),'max_backturn_rad':float(env.max_progress[i]),'takeoff_com_vz':float(env.takeoff_vz[i]),'takeoff_omega':float(env.takeoff_omega[i]),'max_hold_s':float(env.best_hold[i]),'end_gravity':env.robot.data.projected_gravity_b[i].tolist(),'end_quat':env.robot.data.root_quat_w[i].tolist(),'foot_support':env.wcontact[i].tolist(),'roller_support':env.rcontact[i].tolist()})
            forces=env.contacts.data.net_forces_w_history[i].norm(dim=-1).amax(dim=0)
            results[-1].update({'takeoff_momentum':float(env.takeoff_momentum[i]),'airtime_s':float(env.air_max[i]),'peak_com_height_m':float(env.peak_com[i]),'terminal_contact_bodies':[env.contacts.body_names[j] for j in env.bad if float(forces[j])>10.]})
            position=env.robot.data.root_pos_w[i]-env.scene.env_origins[i]
            results[-1].update({'terminal_root_position_m':position.tolist(),'terminal_root_velocity_m_s':env.robot.data.root_lin_vel_w[i].tolist(),'height_failure':bool(position[2]<getattr(env,'minimum_pelvis_height',.18)),'boundary_failure':bool(position[:2].norm()>1.8)})
            first_seen.add(i)
    original_reset(ids)
writer=None;frames=[]
if a.video:
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    writer=imageio.get_writer(str(out/'q1_v2_actual.mp4'),fps=50,codec='libx264',quality=8,macro_block_size=16)
    # Keep the same physical batch and randomization as the independent evaluation.
    # Hide other robots only in the renderer; their physics is unchanged.
    if cfg.scene.num_envs>1:
        import omni.usd
        from pxr import UsdGeom
        stage=omni.usd.get_context().get_stage()
        for index in range(cfg.scene.num_envs):
            if index!=watch:UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/envs/env_{index}/Robot')).MakeInvisible()
seeds=[a.seed] if a.video else [4107,5107,6107]
with torch.inference_mode():
    for seed in seeds:
        current_seed=seed;env._reset_idx=original_reset;env.seed(seed);wrapped.reset();env.metric_history.clear();first_seen=set();env._reset_idx=capture_reset
        obs=wrapped.get_observations()
        for step in range(a.steps if a.video else 310):
            obs,reward,done,extras=wrapped.step(policy(obs))
            if writer:
                frame=env.render();im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,54),fill=(0,0,0))
                draw.text((12,8),'Q1 v2 | single load/push | physical motor limits | candidate demonstration',fill='white')
                draw.text((12,30),f'Attempt {len(results)+1} | episode {float(env._phase()[watch]):.2f}s | backward rotation {float(-env.winding[watch])*180/math.pi:.1f} deg | deterministic policy',fill='white')
                writer.append_data(np.asarray(im))
                if step in [0,50,100,150]:im.save(out/f'frame_{step:03d}.png')
            if not a.video and len(first_seen)==cfg.scene.num_envs:break
        if not a.video and len(first_seen)!=cfg.scene.num_envs:raise RuntimeError('Incomplete first-episode evaluation')
if writer:writer.close()
summary={'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest() if ck else None,'environment_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'deterministic':True,'first_episode_only':not a.video,'num_completed_episodes':len(results),'successes':sum(x['success'] for x in results),'takeoffs':sum(x['real_takeoff'] for x in results),'mean_backward_rotation_rad':sum(x['backward_rotation_rad'] for x in results)/max(1,len(results)),'early_hops':sum(x['early_hop'] for x in results),'mean_max_backturn_rad':sum(x['max_backturn_rad'] for x in results)/max(1,len(results)),'episodes':results}
summary['environment']=environment
summary['environment_module_sha256']=hashlib.sha256(Path('q1_backflip_v2',environment+'.py').read_bytes()).hexdigest()
summary['teacher_sha256']=hashlib.sha256((ck.parent/'teacher.json' if ck else Path(a.teacher)).read_bytes()).hexdigest()
summary['video_env_index']=watch if a.video else None
summary['num_envs']=cfg.scene.num_envs
(out/'evaluation.json').write_text(json.dumps(summary,indent=2));print('EVALUATION_RESULT',json.dumps({k:v for k,v in summary.items() if k!='episodes'}),flush=True)
env.close();app.close()
