import argparse,json,math,time,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=512);p.add_argument('--generations',type=int,default=20);p.add_argument('--out',default='q1_backflip_v2/teacher.json');p.add_argument('--flight_goal',action='store_true');p.add_argument('--jump_goal',action='store_true');p.add_argument('--backflip_goal',action='store_true');p.add_argument('--momentum_goal',action='store_true')
p.add_argument('--environment',choices=['env','compact_env','landing_env','arm_env','live_arm_env'],default='env')
p.add_argument('--replicas',type=int,default=1)
p.add_argument('--landing_goal',action='store_true')
p.add_argument('--checkpoint');p.add_argument('--std-fraction',type=float,default=.3)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
import importlib
module=importlib.import_module(a.environment)
Env,Config,DEFAULT,LOW,HIGH=module.Env,module.Config,module.DEFAULT,module.LOW,module.HIGH
cfg=Config();cfg.scene.num_envs=a.num_envs;cfg.episode_length_s=6. if a.environment in ['arm_env','live_arm_env'] else 4.5;cfg.sim.device=a.device or 'cuda:0';cfg.seed=4107
cfg.sim.physx.gpu_max_rigid_patch_count=2**20
E=Env(cfg);E.search_mode=True
lo=torch.tensor(LOW,device=E.device);hi=torch.tensor(HIGH,device=E.device);mean=torch.tensor(DEFAULT,device=E.device);std=(hi-lo)*a.std_fraction
N=E.num_envs;R=a.replicas
if R<1 or N%R:raise ValueError('num_envs must be divisible by positive replicas')
C=N//R
if C<8:raise ValueError('At least eight independent candidates required')
zero=torch.zeros(N,24,device=E.device);elite_num=min(C,max(8,C//16))
if Path(a.out).exists():
    saved=json.loads(Path(a.out).read_text());mean=torch.tensor(saved['parameters'],device=E.device).clamp(lo,hi)
best=-1.e9;best_parameters=mean.clone();history=[]
launch_parameters=mean.clone()
if a.checkpoint:
    from skills.policy import FrozenActor
    actor=FrozenActor(a.checkpoint,E.device)
else:actor=None
original=E._reset_idx;seen=torch.zeros(N,dtype=torch.bool,device=E.device)
score=torch.full((N,),-10000.,device=E.device);all_metrics=[None]*N

def capture(ids):
    new=ids[~seen[ids]]
    if len(new):
        turn=E.max_progress[new];has=E.has_air[new];pre=E.prehop[new];illegal=E.failed[new];up=-E.robot.data.projected_gravity_b[new,2]
        # Strongly prefer spin in one genuine flight; a high jump alone has negligible value.
        value=25*turn+1.*has.float()+150*E.success[new].float()+5*E.best_hold[new]-25*illegal.float()-150*pre.float()+3*up*(turn>5.8)
        if a.flight_goal:
            value+=has.float()*(5*E.takeoff_omega[new].clamp(0,12)+25*(E.peak_com[new]-E.initial_com_z[new]).clamp(0,.9))
        if a.jump_goal:
            value=has.float()*(100*E.takeoff_vz[new].clamp(0,3)+15*turn+20*(E.peak_com[new]-E.initial_com_z[new]).clamp(0,.9))-40*illegal.float()-150*pre.float()
        if a.backflip_goal:
            quality=(E.takeoff_vz[new]/2.4).clamp(0,1)
            value=has.float()*(30*turn*quality+12*E.takeoff_omega[new].clamp(0,10)*quality+20*E.takeoff_vz[new].clamp(0,3))+150*E.success[new].float()+5*E.best_hold[new]-40*illegal.float()-150*pre.float()
        if a.momentum_goal:
            quality=(E.takeoff_vz[new]/2.4).clamp(0,1)
            value=has.float()*(30*turn*quality+4*E.takeoff_momentum[new].clamp(0,50)*quality+20*E.takeoff_vz[new].clamp(0,3))+150*E.success[new].float()+5*E.best_hold[new]-40*illegal.float()-150*pre.float()
        if a.landing_goal:
            support=torch.where(E.params[new,15]>.5,E.wcontact[new].any(dim=1)&E.rcontact[new].any(dim=1),E.wcontact[new].all(dim=1)).float()
            settled=torch.exp(-E.robot.data.root_ang_vel_b[new].square().sum(dim=1)/4.-E.robot.data.root_lin_vel_w[new,:2].square().sum(dim=1))
            value=15*turn+40*(turn>5.9).float()+60*(turn>5.9).float()*up.clamp(0,1)*support*settled+40*E.best_hold[new]+300*E.success[new].float()-120*illegal.float()-200*pre.float()
        score[new]=value;seen[new]=True
        for i in new.tolist():
            all_metrics[i]={'score':float(score[i]),'max_backturn_rad':float(E.max_progress[i]),'takeoff':bool(E.has_air[i]),'early_hop':bool(E.prehop[i]),'illegal':bool(E.failed[i]),'takeoff_omega':float(E.takeoff_omega[i]),'hold_s':float(E.best_hold[i]),'success':bool(E.success[i]),'end_winding_rad':float(E.winding[i]),'takeoff_com_vz':float(E.takeoff_vz[i]),'takeoff_momentum':float(E.takeoff_momentum[i]),'peak_com_height':float(E.peak_com[i]),'airtime':float(E.air_max[i])}
    original(ids)

for generation in range(a.generations):
    candidates=(mean+std*torch.randn(C,len(mean),device=E.device)).clamp(lo,hi)
    # Keep global winner, default, and random exploration; never discard a physical improvement.
    candidates[0]=best_parameters;candidates[1]=torch.tensor(DEFAULT,device=E.device)
    if generation<5 and not a.checkpoint:
        candidates[C*3//4:]=lo+(hi-lo)*torch.rand(C-C*3//4,len(mean),device=E.device)
    if a.landing_goal and a.environment!='live_arm_env':
        # Preserve the independently validated takeoff; optimize airborne pose and opening.
        ground_indices=[0,1,2,3,4,5,6,7,8,9,14,16,17,18]
        candidates[:,ground_indices]=launch_parameters[ground_indices]
    E.set_parameters(candidates.repeat_interleave(R,dim=0));E._reset_idx=original;E.reset();seen.zero_();score.fill_(-10000);all_metrics=[None]*N;E._reset_idx=capture
    with torch.no_grad():
        for step in range(E.max_episode_length+1):
            E.step(actor(E._get_observations()['policy']) if actor else zero)
            if seen.all():break
    if not seen.all():raise RuntimeError('Search failed to finish all first episodes')
    grouped=score.reshape(C,R)
    robust_score=grouped.mean(dim=1)-.5*grouped.std(dim=1,unbiased=False)
    order=robust_score.argsort(descending=True);elites=candidates[order[:elite_num]]
    mean=.25*mean+.75*elites.mean(dim=0);std=(.35*std+.65*elites.std(dim=0)).clamp_min((hi-lo)*.025)
    candidate_i=int(order[0]);i=candidate_i*R+int(grouped[candidate_i].argmin());local=all_metrics[i]
    replica_metrics=all_metrics[candidate_i*R:(candidate_i+1)*R]
    if float(robust_score[candidate_i])>best:
        best=float(robust_score[candidate_i]);best_parameters=candidates[candidate_i].clone()
        result={'parameters':best_parameters.tolist(),'environment':a.environment,'metrics':local,'replica_metrics':replica_metrics,'robust_score':best,'replicas':R,'generation':generation,'files_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('q1_backflip/env.py'),Path('q1_backflip_v2/env.py'),Path('q1_backflip_v2')/(a.environment+'.py')]},'method':'GPU CEM / physical CubeMars / no external force / standing-only reset'}
        Path(a.out).write_text(json.dumps(result,indent=2))
    entry={'generation':generation,'best':local,'global_score':best,'valid_takeoffs':sum(x['takeoff'] for x in all_metrics),'early_hops':sum(x['early_hop'] for x in all_metrics),'successes':sum(x['success'] for x in all_metrics)}
    history.append(entry);Path(a.out).with_suffix('.search.json').write_text(json.dumps(history,indent=2));print('SEARCH',json.dumps(entry),flush=True)
E.close();app.close()
