"""Independent first-trial validation of the web runtime and model handovers."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--steps',type=int,default=1250);p.add_argument('--seed',type=int,default=4107)
p.add_argument('--variants',default='boot_kneel,supine_elbow,prone_elbow,supine_gripper,prone_gripper')
p.add_argument('--out',required=True);p.add_argument('--checkpoint')
p.add_argument('--unbox-checkpoint')
p.add_argument('--recovery-plant',action='store_true')
p.add_argument('--landing-bank')
p.add_argument('--actor-scales',help='Diagnostic groups only; does not change saved models.')
p.add_argument('--direct-rise',action='store_true',help='Diagnostic landing-specific rise prior.')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from skills.interactive_env import Env,Config
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device or 'cuda:0'
if args.recovery_plant:
    import isaaclab.sim as sim_utils
    material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=.9,friction_combine_mode='multiply',restitution_combine_mode='multiply')
    cfg.sim.physics_material=material;cfg.terrain.physics_material=material
env=Env(cfg,ROOT,args.checkpoint);env.reset();results={}
if args.unbox_checkpoint:
    from skills.policy import FrozenActor
    env.recoveries['landing_rise'].actor=FrozenActor(args.unbox_checkpoint,env.device)
with torch.inference_mode():
    for variant in args.variants.split(','):
        env.seed(args.seed)
        runtime=env.recoveries[variant];env.recovery=runtime
        scales=None
        if args.actor_scales:
            choices=[float(x) for x in args.actor_scales.split(',')]
            scales=torch.tensor([choices[i%len(choices)] for i in range(env.num_envs)],device=env.device)[:,None]
            saved_actor=runtime.actor
            runtime.actor=lambda obs:saved_actor(obs)*scales
        ids=torch.arange(env.num_envs,device=env.device)
        if variant=='boot_kneel':
            # Explicit reset defines this independent test's initial pose. The
            # Unbox button itself must start without writing simulation state.
            runtime.mdp.reset(runtime.context,ids,boot_kneel=True)
            if args.landing_bank:
                bank=json.loads(Path(args.landing_bank).read_text())
                order=[bank['joint_names'].index(n) for n in env.robot.joint_names]
                g=torch.Generator().manual_seed(args.seed)
                indices=torch.randint(len(bank['states']),(env.num_envs,),generator=g).tolist()
                chosen=[bank['states'][i] for i in indices]
                q=torch.tensor([x['joint_position'] for x in chosen],device=env.device)[:,order]
                qd=torch.tensor([x['joint_velocity'] for x in chosen],device=env.device)[:,order]
                root=torch.tensor([x['root_pose']+x['root_velocity'] for x in chosen],device=env.device)
                root[:,:3]+=env.scene.env_origins
                env.robot.write_joint_state_to_sim(q,qd,env_ids=ids)
                env.robot.write_root_pose_to_sim(root[:,:7],env_ids=ids)
                env.robot.write_root_velocity_to_sim(root[:,7:],env_ids=ids)
            # Refresh kinematics and IMU after the explicit test reset, before
            # asking the policy to act. This advances no physical time.
            env.scene.write_data_to_sim();env.sim.forward();env.imu.reset(ids)
            before=env.robot.data.root_pos_w.clone();q_before=env.robot.data.joint_pos.clone()
            assert env.request('unbox')
            runtime=env.recovery
            if args.direct_rise:
                s=runtime.state;s.stage.fill_(4);s.elapsed.zero_();s.hold.zero_()
                s.start_q.copy_(env.robot.data.joint_pos);s.command.copy_(s.start_q)
                s.upright_active.fill_(False);s.stand_sit_back[0]=False
                s.stand_values[0,11]=0.;s.stand_values[0,5]=3.
                s.stand_values[0,6]=45.;s.stand_values[0,7]=8.;s.stand_values[0,12]=6.
                s.dirty=True
            assert torch.equal(before,env.robot.data.root_pos_w)
            assert torch.equal(q_before,env.robot.data.joint_pos)
        else:
            orientation=variant.split('_')[0]
            runtime.place_lie(0 if orientation=='supine' else 1)
            env.mode=orientation
            assert env.request('recovery_'+variant)
        best=torch.zeros(env.num_envs,device=env.device);stood=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        unbroken=torch.ones_like(stood);phase=torch.zeros(env.num_envs,device=env.device)
        trace=[]
        for step in range(args.steps):
            env.step(env.actor_action())
            # Keep the recovery controller active through the validation window;
            # Actual button transitions are tested by play_skills sequences.
            s=runtime.state;s.update()
            local=env.robot.data.root_pos_w-env.scene.env_origins
            unbroken&=(local[:,2]>.08)&(local[:,:2].norm(dim=-1)<3.)
            best=torch.maximum(best,torch.where(unbroken,s.stand_hold,0.))
            stood|=s.stood&unbroken
            phase=torch.maximum(phase,s.stage.float())
            if step%25==0:
                trace.append({'step':step,'stage':int(s.stage[0]),'root':local[0].tolist(),
                    'gravity':env.robot.data.projected_gravity_b[0].tolist(),
                    'q':env.robot.data.joint_pos[0].tolist(),'target':env.idle_target[0].tolist(),
                    'action':runtime.last_action[0].tolist(),'support':s.support_force[0].tolist(),
                    'stand_hold':float(s.stand_hold[0])})
        results[variant]={'seed':args.seed,'runtime_variant':runtime.variant,'landing_bank':args.landing_bank,'trials':env.num_envs,'stand_successes':int(stood.sum()),
            'stable_at_end':int(((runtime.state.stand_hold>=1.)&unbroken).sum()),
            'best_stand_hold_s':best.tolist(),'max_stage':phase.tolist(),
            'unbroken':unbroken.tolist(),'no_pose_writes_at_unbox_handover':variant=='boot_kneel',
            'joint_names':env.robot.joint_names,'position_joint_names':env.names,'trace':trace}
        if scales is not None:
            results[variant]['diagnostic_groups']={str(x):{'trials':int((scales[:,0]==x).sum()),
                'successes':int((stood&(scales[:,0]==x)).sum())} for x in choices}
        print('WEB_SKILL_RESULT',variant,json.dumps({k:v for k,v in results[variant].items() if k not in ['trace','joint_names','position_joint_names']}),flush=True)
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2))
env.close();app.close()
