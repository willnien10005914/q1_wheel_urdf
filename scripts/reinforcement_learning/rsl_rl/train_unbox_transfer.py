"""Fine-tune the existing kneel-to-stand PPO on real backflip landing poses."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=64);p.add_argument('--iterations',type=int,default=30)
p.add_argument('--bank',required=True);p.add_argument('--checkpoint',default='checkpoints/q1_recovery_v3_boot_kneel_ppo.pt')
p.add_argument('--run-name',default='transfer');p.add_argument('--save-interval',type=int,default=50)
p.add_argument('--landing-direct',action='store_true');p.add_argument('--head-scale',type=float,default=1.)
p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--noise-std',type=float)
p.add_argument('--fixed-learning-rate',action='store_true')
p.add_argument('--web-linkage',action='store_true',help='Use the same physical knee-roller linkage as web/backflip.')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch,gymnasium as gym
import isaaclab.sim as sim_utils
import wheel_humanoid_lab.tasks
from wheel_humanoid_lab.tasks.manager_based.recovery_v3.env_cfg import Q1RecoveryV3BootKneelEnvCfg
from wheel_humanoid_lab.tasks.manager_based.recovery_v3.runner_cfg import Q1RecoveryV3BootKneelPPORunnerCfg
from wheel_humanoid_lab.tasks.manager_based.recovery_v3 import mdp
from wheel_humanoid_lab.tasks.manager_based.skate.mdp.actions import RollerMimicActionCfg
from isaaclab.managers import RewardTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from runtime_compat import runner_cfg, load_checkpoint, policy_module, reset_noise_std
from isaaclab.utils.io import dump_yaml
from importlib.metadata import version
bank=json.loads(Path(args.bank).read_text());states=bank['states']
if len(states)<16 or not bank['physical_landings_only']:raise ValueError('Real landing bank required')
cfg=Q1RecoveryV3BootKneelEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=42;cfg.episode_length_s=20.
cfg.sim.device=args.device or 'cuda:0'
# Match the actual web/backflip plant, including contact materials.
cfg.sim.physics_material=sim_utils.RigidBodyMaterialCfg()
cfg.scene.terrain.physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=.9,restitution=0.)
cfg.scene.robot.spawn.usd_dir=str(ROOT/'usd/transfer')
if args.web_linkage:
    cfg.actions.roller_mimic=RollerMimicActionCfg(asset_name='robot',roller_joint_names=['l_knee_roller_joint','r_knee_roller_joint'],knee_joint_names=['l_knee_joint','r_knee_joint'])
    def stable_stand(env):
        s=mdp.state(env);s.update()
        return s.stand_hold.clamp(max=3.)/3.
    cfg.rewards.stable_stand=RewardTermCfg(func=stable_stand,weight=30.)
for name in vars(cfg.events):
    if name!='reset_reference' and not name.startswith('_'):setattr(cfg.events,name,None)
if args.landing_direct:
    mdp.WHEEL_HUMANOID_ROOT_DIR=str(ROOT/'skills/recovery/landing_rise')
cache={}
def reset_bank(env,env_ids,mode=0,boot_kneel=True):
    mdp.reset(env,env_ids,mode=0,boot_kneel=True)
    s=mdp.state(env)
    if not cache:
        order=[bank['joint_names'].index(n) for n in s.robot.joint_names]
        cache['q']=torch.tensor([x['joint_position'] for x in states],device=env.device)[:,order]
        cache['qd']=torch.tensor([x['joint_velocity'] for x in states],device=env.device)[:,order]
        cache['root']=torch.tensor([x['root_pose']+x['root_velocity'] for x in states],device=env.device)
    selected=env_ids if args.landing_direct else env_ids[torch.rand(len(env_ids),device=env.device)<.65]
    if not len(selected):return
    index=torch.randint(len(states),(len(selected),),device=env.device)
    q=cache['q'][index].clone();qd=cache['qd'][index].clone();root=cache['root'][index].clone()
    root[:,:3]+=env.scene.env_origins[selected]
    s.robot.write_joint_state_to_sim(q,qd,env_ids=selected)
    s.robot.write_root_pose_to_sim(root[:,:7],env_ids=selected)
    s.robot.write_root_velocity_to_sim(root[:,7:],env_ids=selected)
    for name in ['start_q','command','held_command','upright_start','upright_command']:getattr(s,name)[selected]=q
    if args.landing_direct:
        s.stage[selected]=4;s.upright_active[selected]=False;s.elapsed[selected]=0.;s.hold[selected]=0.
    s.filtered_force[selected]=0.;s.dirty=True
cfg.events.reset_reference.func=reset_bank
if args.landing_direct:
    def fallen_or_escape(env):
        p=env.scene['robot'].data.root_pos_w-env.scene.env_origins
        return mdp.escaped(env)|((p[:,2]<.15)&(env.episode_length_buf*env.step_dt>.5))
    cfg.terminations.escaped.func=fallen_or_escape
log=ROOT/'logs/unbox_transfer'/(time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name);log.mkdir(parents=True)
env=gym.make('Isaac-Q1-RecoveryV3-BootKneel-v0',cfg=cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
agent=Q1RecoveryV3BootKneelPPORunnerCfg()
agent.max_iterations=args.iterations;agent.save_interval=args.save_interval;agent.device=cfg.sim.device
agent.algorithm.learning_rate=args.learning_rate
if args.fixed_learning_rate:agent.algorithm.schedule='fixed'
runner=OnPolicyRunner(wrapped,runner_cfg(agent),log_dir=str(log),device=cfg.sim.device)
checkpoint=Path(args.checkpoint)
if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
load_checkpoint(runner,checkpoint,load_iteration=False)
runner.current_learning_iteration=0
with torch.no_grad():
    reset_noise_std(runner, args.noise_std if args.noise_std is not None else (.15 if args.landing_direct else .08))
    actor=policy_module(runner)
    mlp=actor.mlp if hasattr(actor,'mlp') else actor.actor
    mlp[-1].weight.mul_(args.head_scale)
    mlp[-1].bias.mul_(args.head_scale)
dump_yaml(str(log/'agent.yaml'),agent)
(log/'manifest.json').write_text(json.dumps({'args':vars(args),'bank_sha256':hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
    'resume_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(Path(mdp.__file__).read_bytes()).hexdigest()},indent=2))
print('TRANSFER_RUN_DIR',str(log),flush=True)
runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
(log/'completed.json').write_text(json.dumps({'iterations':args.iterations,'finite':all(torch.isfinite(p).all().item() for p in policy_module(runner).parameters())}))
env.close();app.close()
