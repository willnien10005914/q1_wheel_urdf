#!/usr/bin/env python3
"""Wait for Q1 leg, then run video-reference PPO, honest recording and delivery.

The waiting/preflight path uses only the standard library and CPU tools. It never
imports CUDA, probes simulator capacity, kills upstream workers, or edits leg data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from reference_campaign_guard import (atomic_json,utcnow,upstream_snapshot,QuietWindow,HandoverLocks,
                                      process_identity,gpu_snapshot)
BASE_CONTRACT='q1-native-manager-v2-23x90'
POLICY_CONTRACT='q1-native-video-v1-23x90'
STOP=False
CHILD=None


def load_json(path):return json.loads(Path(path).read_text())


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def stop_owned_child():
    if CHILD and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass


def interrupted(signum,frame):
    global STOP
    STOP=True;stop_owned_child()


def base_queue():
    spec=importlib.util.spec_from_file_location('reference_native_queue',ROOT/'tools/native_training_queue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def queue_plan(config, smoke=False):
    plan=json.loads(json.dumps(config['queue']))
    if plan.get('contract')!=BASE_CONTRACT or plan.get('num_envs')!=6144:raise ValueError('Campaign must retain 6144-env native layout')
    expected={'standup','kneel','walk','backflip','skate'}
    names=[s['skill'] for s in plan['stages']]
    if len(names)!=len(set(names)) or not {'standup','kneel','backflip'}.issubset(names) or set(names)-expected:
        raise ValueError('Invalid/duplicate campaign skills')
    for stage in plan['stages']:
        if stage.get('iterations')!=10000 or stage.get('policy_contract')!=POLICY_CONTRACT:raise ValueError('Every authorized skill needs 10000 updates and its new policy contract')
        if stage.get('warm_from'):raise ValueError('Do not silently warm-start a new video contract from an unrelated policy')
    if plan.get('chunk_iterations')!=1000:raise ValueError('Validate each 1000-update chunk')
    extra=set(plan.get('source_paths',[]))
    for pattern in config['source_globs']:
        matches=list(ROOT.glob(pattern))
        if not matches:raise FileNotFoundError('Required source glob has no files: '+pattern)
        extra.update(str(p.relative_to(ROOT)) for p in matches if p.is_file())
    extra.update(config['reference']['required_files'])
    extra.update(config.get('source_paths',[]))
    plan['source_paths']=sorted(extra)
    if smoke:
        plan['num_envs']=64;plan['chunk_iterations']=3;plan['evaluation_envs']=8;plan['max_retries']=0
        for stage in plan['stages']:stage['iterations']=3
    return plan


def reference_gate(config):
    """Do not let a placeholder/manual trajectory masquerade as MediaPipe output."""
    report={'ready':False,'reasons':[]}
    try:
        for name in config['reference']['required_files']:
            path=ROOT/name
            if not path.is_file():raise FileNotFoundError(name)
        manifest_path=ROOT/config['reference']['manifest'];manifest=load_json(manifest_path)
        for name,digest in manifest.get('artifacts',{}).items():
            if Path(name).name!=name or sha256(manifest_path.parent/name)!=digest:raise ValueError('Reference artifact SHA mismatch: '+name)
        reference=load_json(ROOT/config['reference']['trajectory'])
        if manifest.get('mediapipe_executed') is not True or manifest.get('approved_for_training') is not True or manifest.get('source_verified') is not True:
            raise ValueError('MediaPipe/source/reference quality approval is incomplete')
        source=reference.get('source',{})
        if source.get('video_id')!='iPItgmg4RBw' or source.get('clip_start_s')!=15. or source.get('clip_end_s')!=23.:raise ValueError('Wrong video segment')
        if reference.get('urdf_sha256')!=sha256(ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf'):raise ValueError('Reference URDF mismatch')
        if source.get('mediapipe_executed') is not True or not source.get('real_clip_sha256'):
            raise ValueError('Reference has no real MediaPipe/clip provenance')
        if reference.get('quality',{}).get('approved_for_training') is not True:raise ValueError('Reference quality not approved')
        report.update(ready=True,reference_sha256=sha256(ROOT/config['reference']['trajectory']),
                      manifest_sha256=sha256(ROOT/config['reference']['manifest']))
    except (OSError,ValueError,KeyError) as error:report['reasons'].append(str(error))
    return report


def seal_models(models, stages, reference_sha, urdf_sha):
    """Seal only CRC-screened queue publications; never promote a preview here."""
    sealed=[]
    for stage in stages:
        skill=stage['skill'];folder=Path(models)/skill
        for metadata_name,weight_name in [('latest.json','latest.pt'),('validated.json','model.pt')]:
            meta_path=folder/metadata_name;weight=folder/weight_name
            try:
                raw=meta_path.read_bytes();meta=json.loads(raw);stamp=weight.stat().st_mtime_ns
                if meta.get('contract')!=stage['policy_contract'] or sha256(weight)!=meta.get('sha256'):continue
                if metadata_name=='validated.json':
                    report_path=(ROOT/meta['validation']).resolve();report_path.relative_to(ROOT.resolve())
                    report=load_json(report_path)
                    expected={'contract':stage['policy_contract'],'skill':skill,'checkpoint_sha256':meta['sha256'],
                              'reference_sha256':reference_sha,'urdf_sha256':urdf_sha,'pass_validation':True}
                    if not meta.get('validated') or any(report.get(k)!=v for k,v in expected.items()):continue
                additions=dict(skill=skill,reference_sha256=reference_sha,urdf_sha256=urdf_sha)
                if all(meta.get(k)==v for k,v in additions.items()):continue
                if weight.stat().st_mtime_ns!=stamp or meta_path.read_bytes()!=raw:continue
                meta.update(additions,sealed_at=utcnow());atomic_json(meta_path,meta);sealed.append(str(meta_path))
            except (OSError,ValueError,KeyError):continue
    return sealed


def run_command(command, log, status, state_path, phase, timeout=None, queue_state=None, seal=None):
    global CHILD
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    status.update(status=phase,updated_at=utcnow(),current_command=[str(c) for c in command],log=str(log.relative_to(ROOT)))
    atomic_json(state_path,status)
    started=time.monotonic()
    with log.open('a') as stream:
        try:CHILD=subprocess.Popen([str(c) for c in command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
        except OSError as error:
            stream.write(str(error)+'\n');return 127
        while CHILD.poll() is None:
            if seal:seal()
            if queue_state and Path(queue_state).exists():
                try:
                    q=load_json(queue_state);status['queue']=dict(status=q.get('status'),current=q.get('current'),effective_num_envs=q.get('effective_num_envs'))
                    # Preserve the user-selected capacity: do not silently accept
                    # the base queue's OOM downscale as a completed 6144 run.
                    if phase=='training' and q.get('effective_num_envs')!=6144:
                        status['error']='Trainer reduced num_envs after a capacity failure; stopping for review'
                        stop_owned_child();CHILD.wait(timeout=45);CHILD=None;return 78
                except (OSError,ValueError):pass
            status['updated_at']=utcnow();atomic_json(state_path,status)
            if STOP or (timeout is not None and time.monotonic()-started>timeout):
                stop_owned_child()
                try:CHILD.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
                CHILD=None;return -signal.SIGTERM if STOP else 124
            time.sleep(5)
        code=CHILD.returncode;CHILD=None
    if seal:seal()
    return code


def smoke_gate(path, stages):
    try:
        state=load_json(path)
        if state.get('status') not in ('completed','completed_with_unvalidated_skills'):return False
        for stage in stages:
            record=state.get('stages',{}).get(stage['skill'],{})
            if record.get('completed_iterations')!=3 or record.get('error') or not record.get('evaluations'):return False
            report=load_json(ROOT/record['evaluations'][-1])
            if report.get('nonfinite')!=0:return False
        return True
    except (OSError,ValueError,KeyError):return False


def safe_gofile_upload(path, out):
    """Persist only public download metadata, never guest/manage tokens."""
    path=Path(path)
    with urllib.request.urlopen('https://api.gofile.io/servers',timeout=60) as response:servers=json.load(response)
    server=servers['data']['servers'][0]['name']
    if not re.fullmatch(r'[A-Za-z0-9-]+',server):raise ValueError('Unexpected upload server')
    boundary='----Q1ReferenceCampaign'+str(time.time_ns())
    header=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: video/mp4\r\n\r\n').encode()
    body=header+path.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    request=urllib.request.Request(f'https://{server}.gofile.io/uploadFile',data=body,
                                  headers={'Content-Type':f'multipart/form-data; boundary={boundary}'},method='POST')
    with urllib.request.urlopen(request,timeout=300) as response:result=json.load(response)
    if result.get('status')!='ok':raise RuntimeError('GoFile upload did not return ok')
    url=result['data']['downloadPage']
    if not re.fullmatch(r'https://gofile\.io/d/[A-Za-z0-9_-]+',url):raise ValueError('Unexpected public download URL')
    public={'name':path.name,'size_bytes':path.stat().st_size,'sha256':sha256(path),'download_page':url,'uploaded_at':utcnow()}
    atomic_json(out,public);return public


def deliver(config, report_dir, model_dir, state, state_path):
    """An isolated Git worktree leaves the user's checkout/index unchanged."""
    work=ROOT/config['output_dir']/'delivery_worktree';branch='delivery/reference-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    def git(directory,*args):return subprocess.run(['git','-C',str(directory),*args],check=True,capture_output=True,text=True,timeout=180).stdout.strip()
    git(ROOT,'fetch','origin','main')
    if work.exists():raise RuntimeError('An earlier delivery worktree needs review: '+str(work))
    git(ROOT,'worktree','add','-b',branch,str(work),'origin/main')
    relatives=[str(Path(report_dir).relative_to(ROOT)),str(Path(model_dir).relative_to(ROOT))]
    for relative in relatives:
        target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/relative,target,dirs_exist_ok=True)
    git(work,'add','-f','--',*relatives);git(work,'commit','-m','Archive video-reference Q1 checkpoints, first-trial validation and GoFile recordings')
    last_error=None
    for attempt in range(3):
        try:
            git(work,'push','origin','HEAD:main')
            state['delivery']={'status':'pushed','commit':git(work,'rev-parse','HEAD'),'at':utcnow()};atomic_json(state_path,state);return
        except subprocess.CalledProcessError as error:
            last_error=str(error);git(work,'fetch','origin','main');git(work,'rebase','origin/main');time.sleep(15)
    raise RuntimeError('GitHub delivery failed after three attempts: '+str(last_error))


def final_validation_gate(checkpoint, report_path, stage, state, exit_code):
    """A failed independent final trial revokes Web access, preserving weights."""
    checkpoint=Path(checkpoint);report_path=Path(report_path)
    try:result=load_json(report_path) if exit_code==0 and report_path.is_file() else {}
    except (OSError,ValueError):result={}
    digest=sha256(checkpoint)
    expected=dict(contract=stage['policy_contract'],skill=stage['skill'],checkpoint_sha256=digest,
                  reference_sha256=state['reference_sha256'],urdf_sha256=state['urdf_sha256'],
                  first_trial_only=True,seed=10424,trials=64,nonfinite=0,pass_validation=True)
    mismatches=[key for key,value in expected.items() if result.get(key)!=value]
    passed=exit_code==0 and not mismatches
    decision=dict(exit_code=exit_code,pass_validation=passed,checkpoint_sha256=digest,
                  preview=checkpoint.name!='model.pt',failed_identity_or_acceptance_fields=mismatches)
    if checkpoint.name=='model.pt':
        path=checkpoint.parent/'validated.json'
        metadata=load_json(path) if path.is_file() else {}
        identity={key:expected[key] for key in ('contract','skill','reference_sha256','urdf_sha256')}
        identity['sha256']=digest
        if any(metadata.get(key)!=value for key,value in identity.items()):
            decision.update(pass_validation=False,metadata_error='Formal model identity does not match final trial')
            return decision
        training_validated=metadata.get('training_validated',metadata.get('validated')) is True
        metadata.update(training_validated=training_validated,validated=bool(passed and training_validated),
                        validation=str(report_path.relative_to(ROOT)),
                        final_validation=dict(checked_at=utcnow(),**decision))
        atomic_json(path,metadata)
        decision['web_enabled']=metadata['validated']
    else:decision['web_enabled']=False
    return decision


def finalize(config, output, models, queue_state, state, state_path, seal):
    reports=ROOT/config['report_dir'];reports.mkdir(parents=True,exist_ok=True)
    state['final_checks']=state.get('final_checks',{})
    for stage in config['queue']['stages']:
        name=stage['skill'];folder=models/name
        checkpoint=folder/'model.pt' if (folder/'model.pt').exists() else folder/'latest.pt'
        if not checkpoint.exists():state['final_checks'][name]={'status':'missing_checkpoint'};continue
        report=reports/(name+'_final_validation.json')
        command=[ROOT/'run_isaac.sh',stage['evaluate_script'],'--skill',name,'--checkpoint',checkpoint,'--num_envs','64',
                 '--steps',str(stage.get('evaluation_steps',1000)),'--seed','10424','--out',report]
        code=run_command(command,output/'logs'/('final_'+name+'.log'),state,state_path,'final_validation',timeout=1200,seal=seal)
        if STOP:return
        state['final_checks'][name]=final_validation_gate(checkpoint,report,stage,state,code)
        atomic_json(state_path,state)
    web_report=reports/'web_motion_validation.json'
    if all(state['final_checks'].get(stage['skill'],{}).get('web_enabled') is True for stage in config['queue']['stages']):
        web_command=[ROOT/'play_skills.sh','--headless','--no-browser','--no-keyboard','--web-port','8796',
                     '--models',models,'--out',web_report,'--campaign-check']
        code=run_command(web_command,output/'logs/final_web.log',state,state_path,'web_motion_validation',timeout=1200,seal=seal)
        if STOP:return
        try:web_result=load_json(web_report) if code==0 else {}
        except (OSError,ValueError):web_result={}
        passed=code==0 and web_result.get('validation',{}).get('pass_validation') is True
        state['web_validation']={'exit_code':code,'pass_validation':passed,'report':str(web_report.relative_to(ROOT))}
        if not passed:
            for stage in config['queue']['stages']:
                path=models/stage['skill']/'validated.json';metadata=load_json(path)
                metadata.update(validated=False,integration_validation=state['web_validation'])
                atomic_json(path,metadata)
                state['final_checks'][stage['skill']]['web_enabled']=False
    else:
        state['web_validation']={'status':'blocked_by_unvalidated_models','pass_validation':False}
    atomic_json(state_path,state)
    record_command=[ROOT/'run_isaac.sh','tools/reference_campaign_record.py','--models',models,
                    '--out',output/'videos','--allow-preview','--headless']
    code=run_command(record_command,output/'logs/record.log',state,state_path,'recording',timeout=3600,seal=seal)
    if STOP:return
    state['recording_exit_code']=code
    if (output/'videos/review.json').exists():shutil.copy2(output/'videos/review.json',reports/'recording.json')
    state['uploads']=state.get('uploads',{})
    for path in sorted((output/'videos').glob('*.mp4')):
        if STOP:return
        out=reports/(path.stem+'.gofile.json')
        try:
            state['uploads'][path.stem]=load_json(out) if out.exists() else safe_gofile_upload(path,out)
        except Exception as error:state['uploads'][path.stem]={'status':'failed','error':str(error)}
        atomic_json(state_path,state)
    seal()
    shutil.copy2(queue_state,reports/'queue_status.json')
    for path in Path(queue_state).parent.glob('*_validation.json'):shutil.copy2(path,reports/path.name)
    for path in (output/'videos').glob('*.png'):shutil.copy2(path,reports/path.name)
    summary={'finished_at':utcnow(),'source_sha256':state['source_sha256'],'reference_sha256':state['reference_sha256'],
             'urdf_sha256':state['urdf_sha256'],'smoke':state.get('smoke'),'skills':state['final_checks'],
             'uploads':state['uploads'],'recording_exit_code':state['recording_exit_code'],
             'web_validation':state['web_validation'],
             'not_retrained':['supine_elbow','prone_elbow','supine_gripper','prone_gripper'],
             'all_motion_validations_passed':state['web_validation']['pass_validation'] and all(v.get('pass_validation') is True for v in state['final_checks'].values()),
             'warning':'A completed iteration budget and a recorded clip are not proof of successful motion.'}
    atomic_json(reports/'summary.json',summary)
    deliver(config,reports,models,state,state_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/reference_campaign_20261024.json')
    p.add_argument('--status',action='store_true');p.add_argument('--check',action='store_true',help='One CPU-only readiness check; never starts Isaac')
    p.add_argument('--capture-upstream',type=Path,help='Write current upstream process/resource identities, without training')
    args=p.parse_args();config=load_json(args.config);output=ROOT/config['output_dir'];state_path=output/'status.json'
    if args.status:print(state_path.read_text() if state_path.exists() else '{"status":"not_started"}');return 0
    if args.capture_upstream:
        snapshot=upstream_snapshot(config['upstream']);atomic_json(args.capture_upstream,snapshot);print(json.dumps(snapshot));return 0
    if args.check:
        report={'upstream':upstream_snapshot(config['upstream']),'reference':reference_gate(config)};print(json.dumps(report,indent=2));return 0 if report['upstream']['ready'] and report['reference']['ready'] else 3
    output.mkdir(parents=True,exist_ok=True)
    own=(output/'campaign.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    state=load_json(state_path) if state_path.exists() else {'campaign':config['campaign'],'started_at':utcnow()}
    if state.get('status')=='completed':print('Reference campaign already completed');return 0
    quiet=QuietWindow(config.get('quiet_seconds',60));locks=None
    while not STOP:
        resource=upstream_snapshot(config['upstream']);reference=reference_gate(config)
        state.update(status='waiting_for_reference' if not reference['ready'] else 'waiting_for_q1_leg',
                     upstream=resource,reference_gate=reference,updated_at=utcnow())
        atomic_json(state_path,state)
        if quiet.observe(resource['ready'] and reference['ready']):
            candidate=HandoverLocks(config['upstream']['lock_file'],config['gpu_lock'])
            if candidate.acquire():
                if upstream_snapshot(config['upstream'])['ready']:locks=candidate;break
                candidate.release()
            quiet.observe(False)
        time.sleep(min(30,config.get('poll_seconds',15)))
    if STOP:state.update(status='stopped',updated_at=utcnow());atomic_json(state_path,state);return 0
    try:
        plan=queue_plan(config);queue=base_queue();source=queue.plan_source_manifest(plan)
        if state.get('source_sha256') and state['source_sha256']!=source:raise RuntimeError('Campaign sources/reference changed; preserve this experiment and use a new state path')
        state.update(source_sha256=source,reference_sha256=reference['reference_sha256'],
                     urdf_sha256=sha256(ROOT/'urdf/q1_inertia/quanta_robot_wheeled.urdf'),resource_handover_at=utcnow())
        atomic_json(state_path,state)
        models=ROOT/config['models_dir'];queue_state=output/'queue/status.json';smoke_state=output/'smoke/status.json'
        full_config=output/'queue/config.json';smoke_config=output/'smoke/config.json'
        atomic_json(full_config,plan);atomic_json(smoke_config,queue_plan(config,smoke=True))
        seal=lambda:seal_models(models,plan['stages'],state['reference_sha256'],state['urdf_sha256'])
        if not smoke_gate(smoke_state,plan['stages']):
            code=run_command([sys.executable,'-u',ROOT/'tools/native_training_queue.py','--config',smoke_config,
                              '--state',smoke_state,'--models',output/'smoke/models'],output/'logs/smoke.log',state,state_path,'smoke',
                             queue_state=smoke_state)
            if STOP:return 0
            if code or not smoke_gate(smoke_state,plan['stages']):raise RuntimeError('64-env smoke did not finish every task without runtime/nonfinite errors')
        web_out=output/'smoke/web_boot.json'
        web_code=run_command([ROOT/'play_skills.sh','--headless','--no-browser','--no-keyboard','--steps','8',
                              '--models',output/'smoke/models','--web-port','8796','--out',web_out],
                             output/'logs/web_boot.log',state,state_path,'web_smoke',timeout=600)
        if STOP:return 0
        web_result=load_json(web_out) if web_code==0 and web_out.exists() else {}
        if web_result.get('final',{}).get('boot_pose')!='video_kneel' or web_result.get('final',{}).get('contract')!=POLICY_CONTRACT:
            raise RuntimeError('Reference Web boot smoke failed; do not launch 6144 envs')
        state['smoke']={'status':'pipeline_passed','num_envs':64,'updates_each':3,'motion_success_required':False}
        full=load_json(queue_state) if queue_state.exists() else {}
        if full.get('status') not in ('completed','completed_with_unvalidated_skills'):
            code=run_command([sys.executable,'-u',ROOT/'tools/native_training_queue.py','--config',full_config,'--state',queue_state,
                              '--models',models],output/'logs/training.log',state,state_path,'training',queue_state=queue_state,seal=seal)
            if STOP:return 0
            if code:raise RuntimeError('Formal training supervisor failed; checkpoints preserved for review')
        state['training_finished_at']=utcnow();atomic_json(state_path,state)
        finalize(config,output,models,queue_state,state,state_path,seal)
        if STOP:return 0
        state.update(status='completed',updated_at=utcnow());atomic_json(state_path,state);return 0
    except Exception as error:
        state.update(status='failed',error=str(error),updated_at=utcnow());atomic_json(state_path,state);print('REFERENCE_CAMPAIGN_FAILED',str(error),flush=True);return 78
    finally:
        if locks:locks.release()
        if STOP:state.update(status='stopped',updated_at=utcnow());atomic_json(state_path,state)

if __name__=='__main__':raise SystemExit(main())
