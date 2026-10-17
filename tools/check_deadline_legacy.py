"""Inspect actual legacy recovery policy dimensions; never execute them on new CAD."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from skills.policy import FrozenActor

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    files={'supine_elbow':'q1_recovery_v3_supine_ppo.pt','prone_elbow':'q1_recovery_v3_prone_ppo.pt',
           'supine_gripper':'q1_recovery_v3_supine_ppo_peak8000.pt','prone_gripper':'q1_recovery_v3_prone_gripper_ppo.pt'}
    results={}
    for name,filename in files.items():
        actor=FrozenActor(ROOT/'checkpoints'/filename,'cpu');actions=actor.mlp[-1].out_features
        results[name]={'status':'requires_native_task_migration','actor_obs_dim':actor.input_dim,'action_dim':actions,
                       'sha256':actor.checkpoint_sha256,'new_urdf_motion_executed':False,
                       'native_90x23_compatible':actor.input_dim==90 and actions==23}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results));return 0
if __name__=='__main__':raise SystemExit(main())
