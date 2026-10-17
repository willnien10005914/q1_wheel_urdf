#!/usr/bin/env python3
"""Deadline-bounded native campaign, final review and isolated GitHub delivery.

Training stops at 04:00 Taipei; final checks and delivery get the remaining time.
A failed skill is reported, never promoted by this tool.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'config/native_deadline_20261019.json'
STATE = ROOT/'outputs/native_deadline_20261019/status.json'
RESULT = ROOT/'docs/training/deadline_20261019/results'
MODELS = ROOT/'checkpoints/native_manager'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def run_checked(command, log, timeout):
    """Bound both process and descendants, including wedged simulator children."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        try:
            proc=subprocess.Popen([str(c) for c in command], cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            stream.write(str(error)+'\n'); return 127
        try: return proc.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            return 124


def finalize():
    plan=json.loads(CONFIG.read_text())
    cutoff=datetime.fromisoformat(plan['delivery_deadline']).timestamp()
    # Reserve at least 30 minutes for artifact transfer/retries.
    review_cutoff=cutoff-1800
    RESULT.mkdir(parents=True, exist_ok=True)
    summary={'review_started_at':datetime.now(timezone.utc).isoformat(), 'checks':{}, 'skills':{},
             'not_new_urdf_trained':['kneel_independent_ppo','supine_elbow','prone_elbow','supine_gripper','prone_gripper'],
             'warning':'Budget completion is not proof of successful motion; see each evaluation.'}
    summary['code_commit']=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    summary['plan']=plan
    logroot=ROOT/'outputs/native_deadline_20261019/final_review'
    def check(name, command, seconds=900):
        if time.time() >= review_cutoff:
            summary['checks'][name]={'status':'not_run_deadline'}; return
        log=logroot/(name+'.log')
        code=run_checked(command,log,min(seconds,review_cutoff-time.time()))
        summary['checks'][name]={'exit_code':code,'status':'passed' if code==0 else 'failed',
                                'log':str(log.relative_to(ROOT))}
        # Preserve bounded diagnostic evidence rather than huge Kit logs.
        (RESULT/(name+'.log')).write_text(log.read_text(errors='replace')[-100000:])
        write_json(RESULT/'summary.json',summary)
    isaac=ROOT/'run_isaac.sh'
    cpu=Path('/home/will/isaacsim/python.sh')
    for pattern in ('test_native_queue.py','test_manager_regression.py','test_native_slide_rewards.py',
                    'test_native_slide_web_gate.py','test_runtime_compat.py','test_deadline_campaign.py','test_skill_controls.py','test_skill_policy.py'):
        check(pattern[:-3],[cpu,'-m','unittest','discover','-s','tests','-p',pattern],300)
    for stage in plan['stages']:
        skill=stage['skill']; folder=MODELS/skill
        path=folder/'model.pt' if (folder/'model.pt').exists() else folder/'latest.pt'
        if not path.exists():
            summary['skills'][skill]={'status':'missing_checkpoint'}; continue
        report=RESULT/(skill+'_final_validation.json')
        report.unlink(missing_ok=True)
        check(skill+'_final_validation',[isaac,stage.get('evaluate_script','scripts/reinforcement_learning/rsl_rl/evaluate_native.py'),
              '--skill',skill,'--checkpoint',path,'--num_envs','64','--steps',str(stage.get('evaluation_steps',500)),
              '--out',report],900)
        summary['skills'][skill]={'checkpoint':str(path.relative_to(ROOT)),
                                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'pass_validation':json.loads(report.read_text()).get('pass_validation') if report.exists() and summary['checks'][skill+'_final_validation'].get('exit_code')==0 else None,
                                 'evaluated_published_best':path.name=='model.pt'}
    check('legacy_recovery_contracts',[cpu,'tools/check_deadline_legacy.py','--out',RESULT/'legacy_recovery_contracts.json'],300)
    # Execute every Web action request, including explicit rejection of legacy Recovery.
    sequence='0:reset,50:stand,100:walk,175:reset,225:kneel,400:unbox,550:reset,600:backflip,900:reset,950:slide,1300:reset,1350:recovery_supine_elbow,1370:recovery_prone_elbow,1390:recovery_supine_gripper,1410:recovery_prone_gripper'
    check('all_web_action_requests',[ROOT/'play_skills.sh','--headless','--no-browser','--no-keyboard',
          '--web-port','8794','--steps','1450','--sequence',sequence,
          '--training-state',STATE,'--out',RESULT/'web_actions.json'],900)
    web_report=RESULT/'web_actions.json'
    if web_report.exists():
        requests=json.loads(web_report.read_text()).get('requests',[])
        recovery={'recovery_supine_elbow','recovery_prone_elbow','recovery_supine_gripper','recovery_prone_gripper'}
        exercised={r['name'] for r in requests}
        rejected={r['name'] for r in requests if not r['accepted']}
        summary['web_request_coverage']={'all_scheduled_actions_requested':{'stand','walk','kneel','unbox','backflip','slide'}.issubset(exercised),
            'four_legacy_recoveries_rejected':recovery.issubset(rejected),'requests':requests}
    # Browser uses a real API and checks gates against actual model availability.
    if time.time() < review_cutoff-180:
        env=dict(os.environ, Q1_DEADLINE_WEB_URL='http://127.0.0.1:8795',
                 Q1_DEADLINE_WEB_OUTPUT=str(RESULT/'web_browser'),
                 PLAYWRIGHT_PACKAGE='/home/will/.cache/ms-playwright-go/1.57.0/package')
        with (logroot/'browser_player.log').open('w') as stream:
            player=subprocess.Popen([str(ROOT/'play_skills.sh'),'--headless','--no-browser','--no-keyboard',
                '--web-port','8795','--steps','15000','--training-state',str(STATE)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                # The helper waits for readiness and runs Chromium itself.
                previous=dict(os.environ);os.environ.update(env)
                try: check('real_browser',['node','tests/verify_deadline_web.cjs'],300)
                finally: os.environ.clear();os.environ.update(previous)
            finally:
                try: os.killpg(player.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: player.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(player.pid,signal.SIGKILL);player.wait()
    # Controller clips cover the two actions without a native independent PPO.
    check('kneel_and_unbox_recording',[isaac,'tools/review_native_actions.py',
          '--stand-checkpoint',MODELS/'stand/model.pt','--smoke-models',MODELS,
          '--cases','kneel','kneel_to_standup','--out',logroot/'videos'],1200)
    if (logroot/'videos/review.json').exists(): shutil.copy2(logroot/'videos/review.json',RESULT/'kneel_unbox_review.json')
    # Include the final queue record and every intermediate quantitative evaluation.
    if STATE.exists():
        shutil.copy2(STATE,RESULT/'queue_status.json')
        for name, record in json.loads(STATE.read_text()).get('stages',{}).items():
            for relative in record.get('evaluations',[]):
                src=ROOT/relative
                if src.is_file(): shutil.copy2(src,RESULT/(name+'_history_'+src.name))
    for report in STATE.parent.glob('*_validation.json'): shutil.copy2(report,RESULT/report.name)
    # Archive both best validated and latest preview, with their separate metadata.
    archive=ROOT/'checkpoints/native_deadline_20261019'; archive.mkdir(parents=True,exist_ok=True)
    for stage in plan['stages']:
        folder=MODELS/stage['skill']
        for name in ('model.pt','validated.json','latest.pt','latest.json'):
            src=folder/name
            if src.is_file():
                dest=archive/stage['skill']/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
    summary['review_finished_at']=datetime.now(timezone.utc).isoformat()
    summary['all_scheduled_motion_validations_passed']=all(summary['skills'].get(s['skill'],{}).get('pass_validation') is True for s in plan['stages'])
    write_json(RESULT/'summary.json',summary)
    # Use an isolated worktree: user edits and their index are never committed.
    delivery={'status':'pending','deadline':plan['delivery_deadline']}
    write_json(logroot/'delivery.json',delivery)
    work=ROOT/'outputs/native_deadline_20261019/delivery_worktree'
    branch='delivery/native-deadline-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    def git(*args): return subprocess.run(['git','-C',str(ROOT),*args],check=True,text=True,capture_output=True,timeout=120).stdout.strip()
    try:
        git('fetch','origin','main')
        git('worktree','add','-b',branch,str(work),'origin/main')
        for relative in ('docs/training/deadline_20261019/results','checkpoints/native_deadline_20261019'):
            dest=work/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(ROOT/relative,dest,dirs_exist_ok=True)
        def wg(*args): return subprocess.run(['git','-C',str(work),*args],check=True,text=True,capture_output=True,timeout=180).stdout.strip()
        wg('add','-f','--','docs/training/deadline_20261019/results','checkpoints/native_deadline_20261019')
        wg('commit','-m','Archive deadline campaign checkpoints and honest all-action validation')
        while time.time()<cutoff:
            try:
                wg('push','origin','HEAD:main')
                delivery.update(status='pushed',commit=wg('rev-parse','HEAD')); break
            except subprocess.CalledProcessError:
                wg('fetch','origin','main'); wg('rebase','origin/main'); time.sleep(15)
        if delivery['status']!='pushed': delivery['status']='failed_deadline'
    except Exception as error: delivery.update(status='failed',error=str(error))
    write_json(logroot/'delivery.json',delivery)
    print('DEADLINE_DELIVERY',json.dumps(delivery),flush=True)
    return 0 if delivery['status']=='pushed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--finalize-only',action='store_true');a=p.parse_args()
    if not a.finalize_only:
        plan=json.loads(CONFIG.read_text())
        timeout=datetime.fromisoformat(plan['training_deadline']).timestamp()-time.time()+1200
        code=run_checked([sys.executable,'-u',ROOT/'tools/native_training_queue.py','--config',CONFIG,'--state',STATE],
                         STATE.parent/'campaign.log',timeout)
        print('DEADLINE_TRAINING_EXIT',code,flush=True)
    return finalize()

if __name__=='__main__': raise SystemExit(main())
