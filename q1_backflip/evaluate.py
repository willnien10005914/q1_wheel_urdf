"""Deterministic first-episode evaluation and honest final-checkpoint recording."""
import argparse,hashlib,json,math
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--video',action='store_true');p.add_argument('--steps',type=int,default=750)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=a.video;app=AppLauncher(a).app
import torch,yaml,numpy as np
from env import Config,Env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
ck=Path(a.checkpoint).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ck.parent/'manifest.json').read_text())
source=Path('q1_backflip/env.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files']['q1_backflip/env.py'],'Environment differs from training snapshot'
cfg=Config();cfg.scene.num_envs=1 if a.video else 32;cfg.seed=42;cfg.scene.env_spacing=40. if a.video else 4.
cfg.sim.device=a.device or 'cuda:0';cfg.viewer.origin_type='env';cfg.viewer.env_index=0;cfg.viewer.eye=(3.,-4.,2.6);cfg.viewer.lookat=(0,0,1.1);cfg.viewer.resolution=(960,720)
env=Env(cfg,render_mode='rgb_array' if a.video else None);env.updates=20000
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
agent=yaml.safe_load((ck.parent/'agent.yaml').read_text());runner=OnPolicyRunner(wrapped,agent,log_dir=None,device=cfg.sim.device);runner.load(str(ck));policy=runner.get_inference_policy()
results=[];first_seen=set();current_seed=42;original_reset=env._reset_idx
# Capture terminal telemetry BEFORE reset clears evidence.
def capture_reset(ids):
    if ids is None:ids=env.robot._ALL_INDICES
    for i in ids.tolist():
        if a.video or i not in first_seen:
            results.append({'seed':current_seed,'env':i,'success':bool(env.success[i]),'real_takeoff':bool(env.jumped[i]),'backward_rotation_rad':float(-env.winding[i]),'peak_pelvis_height_m':float(env.peak[i]),'illegal_contact':bool(env.failed[i]),'landing_hold_s':float(env.hold[i]),'duration_s':float(env.episode_length_buf[i])*env.step_dt})
            first_seen.add(i)
    original_reset(ids)
writer=None;frames=[]
if a.video:
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    writer=imageio.get_writer(str(out/'q1_20k_actual.mp4'),fps=50,codec='libx264',quality=8,macro_block_size=16)
seeds=[42] if a.video else [4107,5107,6107]
with torch.inference_mode():
    for seed in seeds:
        current_seed=seed;env._reset_idx=original_reset;env.seed(seed);wrapped.reset();env.metric_history.clear();first_seen=set();env._reset_idx=capture_reset
        obs=wrapped.get_observations()
        for step in range(a.steps if a.video else 260):
            obs,reward,done,extras=wrapped.step(policy(obs))
            if writer:
                frame=env.render();im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,54),fill=(0,0,0))
                draw.text((12,8),'Q1 | 6144 environments | 20k checkpoint | NOT a completed backflip',fill='white')
                draw.text((12,30),f'Attempt {len(results)+1} | episode {float(env._phase()[0]):.2f}s | backward rotation {float(-env.winding[0])*180/math.pi:.1f} deg | deterministic policy',fill='white')
                writer.append_data(np.asarray(im))
                if step in [0,50,100,150]:im.save(out/f'frame_{step:03d}.png')
            if not a.video and len(first_seen)==cfg.scene.num_envs:break
        if not a.video and len(first_seen)!=cfg.scene.num_envs:raise RuntimeError('Incomplete first-episode evaluation')
if writer:writer.close()
summary={'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'environment_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'deterministic':True,'first_episode_only':not a.video,'num_completed_episodes':len(results),'successes':sum(x['success'] for x in results),'takeoffs':sum(x['real_takeoff'] for x in results),'mean_backward_rotation_rad':sum(x['backward_rotation_rad'] for x in results)/max(1,len(results)),'episodes':results}
(out/'evaluation.json').write_text(json.dumps(summary,indent=2));print('EVALUATION_RESULT',json.dumps({k:v for k,v in summary.items() if k!='episodes'}),flush=True)
env.close();app.close()
