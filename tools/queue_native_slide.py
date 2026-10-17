"""Wait behind the current native queue, then train sliding with its own state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
state=ROOT/'outputs/native_slide_queue/waiting.json'
state.parent.mkdir(parents=True,exist_ok=True)
upstream=ROOT/'outputs/native_manager/queue/status.json'
while True:
    try:status=json.loads(upstream.read_text())['status']
    except (OSError,ValueError,KeyError):status='not_started'
    state.write_text(json.dumps(dict(status='waiting_for_native_queue',upstream_status=status,
                                    num_envs=6144,skill='slide',iterations=20000))+'\n')
    if status in ('completed','completed_with_unvalidated_skills'):break
    if status=='failed':
        state.write_text(json.dumps(dict(status='blocked_upstream_failed',skill='slide'))+'\n')
        raise SystemExit(78)
    time.sleep(10)
os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'tools/native_training_queue.py'),
    '--config',str(ROOT/'config/native_slide_training_6144.json'),
    '--state',str(ROOT/'outputs/native_slide_queue/status.json')])
