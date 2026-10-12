"""Deterministic inference for both legacy ActorCritic and rsl-rl 5 checkpoints."""
from pathlib import Path
import hashlib
import torch
from torch import nn

class FrozenActor(nn.Module):
    def __init__(self, path, device):
        super().__init__()
        payload=torch.load(Path(path),map_location=device,weights_only=False)
        self.checkpoint_path=str(Path(path).resolve())
        self.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.training_iteration=payload.get('iter')
        if 'actor_state_dict' in payload:
            state=payload['actor_state_dict'];prefix='mlp.';normal='obs_normalizer.'
        else:
            state=payload['model_state_dict'];prefix='actor.';normal='actor_obs_normalizer.'
        weights=sorted((int(k[len(prefix):].split('.')[0]),v) for k,v in state.items() if k.startswith(prefix) and k.endswith('.weight'))
        layers=[]
        for index,weight in weights:
            while len(layers)<index:layers.append(nn.ELU())
            layer=nn.Linear(weight.shape[1],weight.shape[0],device=device)
            layer.weight.data.copy_(weight);layer.bias.data.copy_(state[f'{prefix}{index}.bias'])
            layers.append(layer)
        self.mlp=nn.Sequential(*layers)
        self.input_dim=weights[0][1].shape[1]
        self.register_buffer('mean',state.get(normal+'_mean',torch.zeros(1,self.input_dim,device=device)))
        self.register_buffer('std',state.get(normal+'_std',torch.ones(1,self.input_dim,device=device)))
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self,obs):
        if obs.shape[-1]!=self.input_dim:raise ValueError(f'Expected {self.input_dim} observations; got {obs.shape[-1]}')
        # rsl-rl EmpiricalNormalization default epsilon is 0.01. Matching it
        # matters especially for the near-constant gripper/neck channels.
        return self.mlp((obs-self.mean)/(self.std+.01))
