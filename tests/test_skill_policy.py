"""Inference must match the training actor, including empirical normalization."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from skills.policy import FrozenActor

for path in [ROOT/'checkpoints/q1_recovery_v3_supine_ppo.pt',ROOT/'checkpoints/backflip_v2_20k/model_19999.pt',ROOT/'checkpoints/backflip_arm_rest/model.pt',ROOT/'checkpoints/q1_recovery_v3_prone_gripper_ppo.pt',ROOT/'checkpoints/q1_unbox_transfer_ppo.pt',ROOT/'checkpoints/q1_unbox_landing_ppo.pt']:
    actor=FrozenActor(path,'cpu')
    payload=torch.load(path,weights_only=False,map_location='cpu')
    if 'actor_state_dict' in payload:state=payload['actor_state_dict']
    else:
        old=payload['model_state_dict'];state={}
        for key,value in old.items():
            if key.startswith('actor.'):state['mlp.'+key[len('actor.'):]]=value
            elif key.startswith('actor_obs_normalizer.'):state['obs_normalizer.'+key[len('actor_obs_normalizer.'):]]=value
        state['distribution.std_param']=old['std']
    obs=TensorDict({'policy':torch.randn(32,actor.input_dim)},batch_size=[32])
    reference=MLPModel(obs,obs_groups={'actor':['policy']},obs_set='actor',output_dim=24,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'rsl_rl.modules.GaussianDistribution','init_std':.05,'std_type':'scalar'})
    reference.load_state_dict(state);reference.eval()
    with torch.inference_mode():
        expected=reference(obs);actual=actor(obs['policy'])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    print('POLICY_PARITY_PASS',path.name,float((actual-expected).abs().max()))
