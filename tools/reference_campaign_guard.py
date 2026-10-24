"""CPU-only cooperative gate for the Q1 leg -> wheeled-Q1 GPU handover.

No CUDA imports, benchmark, signal delivery or upstream file mutation occurs here.
The upstream supervisor's existing flock is acquired read-only after completion.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TERMINAL_UPSTREAM = frozenset(('passed', 'complete', 'budget_exhausted'))


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def process_identity(pid, proc_root=Path('/proc')):
    """PID plus Linux birth tick prevents recycled PID confusion."""
    directory=Path(proc_root)/str(pid)
    try:
        tail=(directory/'stat').read_text().rsplit(')',1)[1].split()
        raw=(directory/'cmdline').read_bytes()
        return dict(pid=int(pid),start_ticks=int(tail[19]),session_id=int(tail[3]),
                    cmdline_sha256=hashlib.sha256(raw).hexdigest(),
                    cmdline=raw.replace(b'\0',b' ').decode(errors='replace').strip(),
                    cwd=os.readlink(directory/'cwd'))
    except FileNotFoundError:return None
    # Permission/parse errors deliberately propagate; they do not mean idle.


def process_exists(identity, proc_root=Path('/proc')):
    observed=process_identity(identity['pid'],proc_root)
    return bool(observed and observed['start_ticks']==identity['start_ticks'])


def scan_upstream_processes(repo, proc_root=Path('/proc')):
    """Find replacement supervisor/chunk workers even when original PID exited."""
    found=[];repo=str(Path(repo).resolve())
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():continue
        try:
            raw=(directory/'cmdline').read_bytes()
            command=raw.replace(b'\0',b' ').decode(errors='replace').strip()
            if not command:continue
            executable=Path(command.split()[0]).name.lower()
            if 'python' not in executable:continue
            if not any(token in command for token in ('train.py','training.py','run_upper_body_training.py','rsl_rl/train')):continue
            identity=process_identity(int(directory.name),proc_root)
            if not identity:continue
            if repo in command or identity['cwd']==repo:found.append(identity)
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            # A process of another user may be unreadable. GPU compute-app query
            # still blocks any such GPU consumer; do not inspect their secrets.
            if directory.stat().st_uid==os.getuid():raise
    return found


def _run(command):
    return subprocess.run(command,check=True,capture_output=True,text=True,timeout=20).stdout.strip()


def service_state(name, runner=_run):
    try:value=runner(['systemctl','--user','show',name,'--property=ActiveState','--value']).strip()
    except subprocess.CalledProcessError:
        listing=runner(['systemctl','--user','list-units','--all','--type=service','--plain','--no-legend','--no-pager'])
        if not any(name in line.split()[:2] for line in listing.splitlines()):return 'inactive'
        raise
    if value not in ('inactive','failed','active','activating','deactivating','reloading'):raise RuntimeError('Unknown service state '+repr(value))
    return value


def gpu_snapshot(index=0, runner=_run):
    listing=runner(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'])
    gpus=[]
    for line in listing.splitlines():
        if not line.strip():continue
        parts=[p.strip() for p in line.split(',')]
        if len(parts)!=6:raise RuntimeError('Unparseable nvidia-smi GPU listing')
        gpus.append(dict(index=int(parts[0]),uuid=parts[1],name=parts[2],memory_total_mib=int(parts[3]),
                         memory_used_mib=int(parts[4]),utilization_percent=int(parts[5])))
    selected=next((g for g in gpus if g['index']==int(index)),None)
    if selected is None:raise RuntimeError('Configured GPU not visible')
    apps=[]
    output=runner(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                   '--format=csv,noheader,nounits'])
    for line in output.splitlines():
        if not line.strip():continue
        parts=[p.strip() for p in line.split(',')]
        if len(parts)!=4:raise RuntimeError('Unparseable nvidia-smi compute-app listing')
        if parts[0]==selected['uuid']:
            apps.append(dict(gpu_uuid=parts[0],pid=int(parts[1]),name=parts[2],memory_mib=parts[3]))
    return dict(gpus=gpus,selected_gpu=selected,compute_apps=apps,
                parallel_training_enabled=False,
                reason='Single shared GPU: no memory/compute isolation; do not benchmark or co-run a 6144-env job.')


def upstream_snapshot(config, runner=_run, proc_root=Path('/proc')):
    observations={'checked_at':utcnow(),'ready':False,'reasons':[],'status_files':[],'services':{},'processes':[]}
    try:
        paths=config.get('status_files',[])
        follow=config.get('follow_latest_run')
        if follow:
            latest=json.loads(Path(follow['status_path']).read_text())
            if latest.get('robot')!=follow['expected_robot']:raise RuntimeError('Unexpected upstream robot identity')
            run=Path(latest['run']).resolve();run.relative_to(Path(follow['run_root']).resolve())
            observations['tracked_run']=str(run)
            paths=[dict(path=follow['status_path'],expected_run=str(run)),dict(path=str(run/'supervisor_state.json'),expected_run=str(run))]
        if not paths:raise RuntimeError('Upstream status_files must be configured')
        for entry in paths:
            path=Path(entry['path']);value=json.loads(path.read_text())
            key=entry.get('state_key','state');state=value.get(key)
            row={'path':str(path),'state':state,'run':value.get('run'),'updated_at':value.get('updated_at')}
            observations['status_files'].append(row)
            expected=entry.get('expected_run')
            if expected and value.get('run')!=expected:observations['reasons'].append('Upstream run identity changed: '+str(path))
            if state not in entry.get('terminal_states',sorted(TERMINAL_UPSTREAM)):
                observations['reasons'].append('Upstream supervisor not terminal: '+str(state))
        services=config.get('services',[])
        if not services:raise RuntimeError('Upstream service identity must be configured')
        for name in services:
            state=service_state(name,runner);observations['services'][name]=state
            if state not in ('inactive','failed'):observations['reasons'].append('Upstream service remains '+state+': '+name)
        for identity in config.get('captured_processes',[]):
            if process_exists(identity,proc_root):observations['processes'].append(identity)
        for identity in scan_upstream_processes(config['repo'],proc_root):
            if not any(i['pid']==identity['pid'] for i in observations['processes']):observations['processes'].append(identity)
        if observations['processes']:observations['reasons'].append('Upstream training/supervisor process still exists')
        observations['gpu']=gpu_snapshot(config.get('gpu_index',0),runner)
        if observations['gpu']['compute_apps']:observations['reasons'].append('Configured GPU still has compute consumers')
        observations['ready']=not observations['reasons']
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
        observations['reasons'].append('Cannot establish safe resource handover: '+str(error))
    return observations


class HandoverLocks:
    """Own both the upstream's real lock and our shared GPU lock until release."""
    def __init__(self, upstream_lock, gpu_lock):
        self.upstream_lock=Path(upstream_lock);self.gpu_lock=Path(gpu_lock);self.streams=[]

    def acquire(self):
        try:
            # Never truncate/create the other project's lock or change its owner.
            upstream=self.upstream_lock.open('r');self.streams.append(upstream)
            fcntl.flock(upstream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
            shared=self.gpu_lock.open('a+');self.streams.append(shared)
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except (OSError,BlockingIOError):self.release();return False

    def release(self):
        while self.streams:
            stream=self.streams.pop()
            try:fcntl.flock(stream,fcntl.LOCK_UN)
            finally:stream.close()


@dataclass
class QuietWindow:
    seconds:float=60.
    idle_since:float|None=None

    def observe(self, ready, now=None):
        now=time.monotonic() if now is None else now
        if not ready:self.idle_since=None;return False
        if self.idle_since is None:self.idle_since=now
        return now-self.idle_since>=self.seconds
