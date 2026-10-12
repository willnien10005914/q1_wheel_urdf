"""Record preserved checkpoint hashes, iterations and formats without simulation."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
models=[]
for path in sorted((ROOT/'checkpoints').rglob('*.pt')):
    if 'exported' in path.parts:continue
    state=torch.load(path,map_location='cpu',weights_only=False)
    models.append({'file':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'iteration':state.get('iter'),
        'format':'rsl_rl5' if 'actor_state_dict' in state else 'legacy_actor_critic'})
(ROOT/'docs/training/checkpoint_inventory.json').write_text(json.dumps({'models':models,
    'scope':'Effective, selected peak and final historical checkpoints; intermediate save intervals remain in local logs.'},indent=2))
print('INDEXED_MODELS',len(models),'BYTES',sum(m['bytes'] for m in models))
