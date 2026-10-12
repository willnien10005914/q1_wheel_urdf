import argparse,json,os,time,hashlib,shutil
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=64);p.add_argument('--max_iterations',type=int,default=5);p.add_argument('--run_name',default='smoke');p.add_argument('--resume');p.add_argument('--warm_start',action='store_true');p.add_argument('--landing',choices=['wheels','kneel'],default='wheels');p.add_argument('--seed',type=int,default=42);p.add_argument('--evaluate_steps',type=int,default=0);p.add_argument('--curriculum_offset',type=int,default=0)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from env import Config,Env
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,RslRlOnPolicyRunnerCfg,RslRlMLPModelCfg,RslRlPpoAlgorithmCfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.landing=args.landing
cfg.sim.device=args.device or 'cuda:0'
cfg.sim.physx.gpu_max_rigid_patch_count=2**20
cfg.sim.physx.gpu_max_rigid_contact_count=2**23
log=Path('logs/q1_backflip')/(time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name);log.mkdir(parents=True)
agent=RslRlOnPolicyRunnerCfg(num_steps_per_env=24,max_iterations=args.max_iterations,save_interval=200,experiment_name='q1_backflip',obs_groups={'actor':['policy'],'critic':['policy']},clip_actions=None,
actor=RslRlMLPModelCfg(hidden_dims=[512,256,128],activation='elu',obs_normalization=True,distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=.25,std_type='scalar')),
critic=RslRlMLPModelCfg(hidden_dims=[512,256,128],activation='elu',obs_normalization=True),
algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.002,num_learning_epochs=5,num_mini_batches=4,learning_rate=3.e-4,schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=.5))
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
import importlib.metadata
agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version("rsl-rl-lib"))
agent.device=cfg.sim.device;agent.seed=args.seed
manifest={'args':vars(args),'created':time.strftime('%Y-%m-%d %H:%M:%S'),'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('q1_backflip').glob('*.py')},'original_q1_root':os.environ['WHEEL_HUMANOID_ROOT'],'pid':os.getpid(),'rsl_rl_version':importlib.metadata.version('rsl-rl-lib')}
for dependency in ['urdf/wheel_humanoid.urdf','source/wheel_humanoid_lab/wheel_humanoid_lab/assets/wheel_humanoid.py','source/wheel_humanoid_lab/wheel_humanoid_lab/assets/q1_spec.py','source/wheel_humanoid_lab/wheel_humanoid_lab/actuators/cubemars.py']:
    f=Path(os.environ['WHEEL_HUMANOID_ROOT'])/dependency
    manifest['files'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
motion=Path('reference_isaacsim_backflip/motions/backflip_g1.npz')
manifest['files'][str(motion)]=hashlib.sha256(motion.read_bytes()).hexdigest()
(log/'manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copytree('q1_backflip',log/'source_snapshot',ignore=shutil.ignore_patterns('usd','__pycache__','*.npz'))
env=Env(cfg); wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(log),device=cfg.sim.device)
if args.resume:
    runner.load(args.resume)
    runner.current_learning_iteration=0 if args.warm_start else runner.current_learning_iteration+1
# Curriculum advances at the start of each rollout, using completed runner iteration.
orig=env._pre_physics_step
def before(actions):
    env.updates=runner.current_learning_iteration+args.curriculum_offset
    orig(actions)
env._pre_physics_step=before
obs=wrapped.get_observations()
if not torch.isfinite(obs['policy']).all():raise RuntimeError('initial observations nonfinite')
dump_yaml(str(log/'env.yaml'),cfg);dump_yaml(str(log/'agent.yaml'),agent)
(log/'running.json').write_text(json.dumps({'pid':os.getpid(),'start_iteration':runner.current_learning_iteration,'target_iterations':args.max_iterations,'num_envs':args.num_envs,'landing':args.landing,'state':'running'},indent=2))
print('RUN_DIR',str(log.resolve()),'START_ITER',runner.current_learning_iteration,flush=True)
if args.evaluate_steps:
    policy=runner.get_inference_policy(); wrapped.reset()
    with torch.inference_mode():
        for _ in range(args.evaluate_steps):
            obs=wrapped.get_observations();wrapped.step(policy(obs))
    metrics=env.metric_history
else:
    remaining=args.max_iterations-runner.current_learning_iteration
    if remaining<=0:raise ValueError("total iteration budget already reached")
    runner.learn(num_learning_iterations=remaining,init_at_random_ep_len=False)
    metrics=env.metric_history
finite_parameters=all(torch.isfinite(v).all().item() for model in [runner.alg.actor,runner.alg.critic] for v in model.parameters())
if not finite_parameters:raise FloatingPointError('nonfinite learned parameters')
summary={'iterations':runner.current_learning_iteration,'num_envs':args.num_envs,'finite':finite_parameters,'episodes_batches':len(metrics),'latest_metrics':metrics[-10:]}
(log/'running.json').write_text(json.dumps({'pid':os.getpid(),'state':'completed','last_iteration':runner.current_learning_iteration}))
(log/'completed.json').write_text(json.dumps(summary,indent=2));print('TRAINING_COMPLETED',json.dumps(summary),flush=True)
env.close();app.close()
