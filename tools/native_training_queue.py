#!/usr/bin/env python3
"""Persistent, single-GPU sequential manager PPO queue with checkpoint resume.

This supervisor needs only the Python standard library. Isaac children are
started through run_isaac.sh. Evaluation and training never overlap each other.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CONTRACT="q1-native-manager-v2-23x90"
STOP=False
CHILD=None
STATE_PATH=None
CHECKPOINT_CACHE={}
TERMINAL_FAILURE=78  # systemd must not restart exhausted retries/config errors.
STALL_TIMEOUT=600.  # A live supervisor heartbeat is not simulator progress.


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)


def source_manifest():
    paths=list((ROOT/"source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/inertia").rglob("*.py"))
    paths += list((ROOT/"source/wheel_humanoid_lab/wheel_humanoid_lab/actuators").glob("*.py"))
    paths += [ROOT/"source/wheel_humanoid_lab/wheel_humanoid_lab/assets/q1_inertia.py",
              ROOT/"urdf/q1_inertia/quanta_robot_wheeled.urdf",
              ROOT/"scripts/reinforcement_learning/rsl_rl/train.py",
              ROOT/"scripts/reinforcement_learning/rsl_rl/runtime_compat.py",
              ROOT/"scripts/reinforcement_learning/rsl_rl/evaluate_native.py",
              ROOT/"source/wheel_humanoid_lab/wheel_humanoid_lab/assets/wheel_humanoid.py",
              ROOT/"source/wheel_humanoid_lab/wheel_humanoid_lab/assets/q1_spec.py",
              ROOT/"config/q1_wheel_components.yaml"]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def latest_checkpoint(directory,prefix):
    candidates=list(directory.glob(f"*{prefix}*/model_*.pt"))
    candidates=[p for p in candidates if p.stat().st_size>1024 and time.time()-p.stat().st_mtime>2.]
    for path in sorted(candidates,key=lambda p:(int(re.search(r"model_(\d+)",p.name)[1]),p.stat().st_mtime),reverse=True):
        key=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
        if key not in CHECKPOINT_CACHE:
            try:
                with zipfile.ZipFile(path) as archive:
                    CHECKPOINT_CACHE[key]=archive.testzip() is None and any(n.endswith("data.pkl") for n in archive.namelist())
            except (OSError,zipfile.BadZipFile):CHECKPOINT_CACHE[key]=False
        if CHECKPOINT_CACHE[key]:return path
    return None


def publish(source,folder,validated=False,validation=None):
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/("model.pt" if validated else "latest.pt")
    tmp=target.with_suffix(".tmp");shutil.copy2(source,tmp);tmp.replace(target)
    metadata=dict(contract=CONTRACT,manager_based=True,new_urdf=True,validated=validated,
                  source=str(source.relative_to(ROOT)),iteration=int(re.search(r"model_(\d+)",source.name)[1]),
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),validation=validation)
    atomic_json(folder/("validated.json" if validated else "latest.json"),metadata)


def interrupted(signum,frame):
    global STOP
    STOP=True
    if CHILD and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass


def run_child(command,log,status,state_path,stage=None,prefix=None,models=None):
    global CHILD
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as stream:
        environment=dict(os.environ,OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
        if status["current"].get("fresh_warm_start"):environment["RESET_NOISE_STD"]="0.20"
        CHILD=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                               env=environment,start_new_session=True)
        last_output=time.monotonic();last_size=0
        status["current"]["pid"]=CHILD.pid;status["current"]["log"]=str(log.relative_to(ROOT))
        last_published=None
        while CHILD.poll() is None:
            size=log.stat().st_size
            if size!=last_size:
                last_size=size;last_output=time.monotonic()
            if time.monotonic()-last_output>STALL_TIMEOUT:
                status["current"]["watchdog_error"]=f"Child produced no log output for {STALL_TIMEOUT:g}s"
                atomic_json(state_path,status)
                print("QUEUE_STALL",status["current"]["watchdog_error"],str(log),flush=True)
                try:os.killpg(CHILD.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
                CHILD=None
                return 124  # Goes through the existing bounded retry budget.
            status["updated_at"]=datetime.now(timezone.utc).isoformat()
            if stage:
                checkpoint=latest_checkpoint(ROOT/"logs/rsl_rl"/f"q1_native_{stage['skill']}",prefix)
                if checkpoint:
                    completed=int(re.search(r"model_(\d+)",checkpoint.name)[1])+1
                    status["current"]["completed_iterations"]=completed
                    if str(checkpoint)!=last_published:
                        publish(checkpoint,models/stage["skill"])
                        last_published=str(checkpoint)
            atomic_json(state_path,status)
            if STOP:
                try:CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
                return -signal.SIGTERM
            time.sleep(5)
        result=CHILD.returncode;CHILD=None
    return result


def main():
    global STATE_PATH,STALL_TIMEOUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"config/native_training_queue.json")
    p.add_argument("--state",type=Path,default=ROOT/"outputs/native_manager/queue/status.json")
    p.add_argument("--models",type=Path,default=ROOT/"checkpoints/native_manager")
    p.add_argument("--status",action="store_true")
    p.add_argument("--stall-timeout",type=float,default=600.,help="Fail and retry a child with no log output for this many seconds")
    args=p.parse_args()
    if args.stall_timeout<30: p.error("--stall-timeout must be at least 30 seconds")
    STALL_TIMEOUT=args.stall_timeout
    args.config=args.config.resolve();args.state=args.state.resolve();args.models=args.models.resolve()
    STATE_PATH=args.state
    if args.status:
        print(args.state.read_text() if args.state.exists() else '{"status":"not_started"}');return 0
    plan=json.loads(args.config.read_text())
    if plan["contract"]!=CONTRACT:raise ValueError("Wrong queue contract")
    if plan["num_envs"]<4 or plan["chunk_iterations"]<1:raise ValueError("Invalid queue sizes")
    args.state.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.state.parent/"queue.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=source_manifest()
    if args.state.exists():
        status=json.loads(args.state.read_text())
        if status["source_sha256"]!=source or status["plan"]!=plan:
            raise RuntimeError("Queue source or plan changed; use a new --state path for a new experiment")
    else:
        status=dict(contract=CONTRACT,run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                    plan=plan,source_sha256=source,stages={},effective_num_envs=plan["num_envs"],current={})
    status["status"]="running"
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    for index,stage in enumerate(plan["stages"]):
        name=stage["skill"]
        record=status["stages"].setdefault(name,dict(completed_iterations=0,evaluations=[],attempts=0))
        prefix=f"native_{status['run_id']}_{name}_"
        directory=ROOT/"logs/rsl_rl"/f"q1_native_{name}"
        while not STOP:
            if source_manifest()!=source:raise RuntimeError("Controller sources changed during active training")
            checkpoint=latest_checkpoint(directory,prefix)
            if checkpoint:record["completed_iterations"]=max(record["completed_iterations"],int(re.search(r"model_(\d+)",checkpoint.name)[1])+1)
            done=record["completed_iterations"]
            pending_evaluation=done>record.get("last_evaluated_iteration",0)
            if done>=stage["iterations"] and not pending_evaluation:break
            if not pending_evaluation and record["attempts"]>plan["max_retries"]:
                status["status"]="failed"
                status["error"]=f"Retry budget exhausted for {name}; inspect logs before starting a new experiment"
                atomic_json(args.state,status);return TERMINAL_FAILURE
            count=min(plan["chunk_iterations"],stage["iterations"]-done)
            log=ROOT/"logs/native_manager_queue"/status["run_id"]/f"{name}_{done:06d}.log"
            status["current"]=dict(index=index,skill=name,task=stage["task"],phase="training",
                                   completed_iterations=done,total_iterations=stage["iterations"],fresh_warm_start=False)
            command=[str(ROOT/"run_isaac.sh"),"scripts/reinforcement_learning/rsl_rl/train.py","--headless",
                     "--task",stage["task"],"--num_envs",str(status["effective_num_envs"]),"--max_iterations",str(count),
                     "--seed",str(plan["seed"]),"--run_name",prefix+str(done),"--experiment_name",f"q1_native_{name}",
                     "--no_publish_checkpoint",f"env.curriculum.skill.params.start_iter={done}"]
            if checkpoint:command += ["--resume","--checkpoint",str(checkpoint),"--resume_next_iteration"]
            elif stage.get("warm_from"):
                warm=args.models/stage["warm_from"]
                warm=warm/"model.pt" if (warm/"model.pt").exists() else warm/"latest.pt"
                if warm.is_file():
                    command += ["--resume","--checkpoint",str(warm),"--reset_iteration"]
                    status["current"]["fresh_warm_start"]=True
            if not pending_evaluation:
                atomic_json(args.state,status)
                print("QUEUE_TRAIN",name,done,count,flush=True)
                code=run_child(command,log,status,args.state,stage,prefix,args.models)
                if STOP:break
                if code:
                    record["attempts"]+=1
                    text=log.read_text(errors="replace").lower()
                    if "out of memory" in text:status["effective_num_envs"]=max(64,status["effective_num_envs"]//2)
                    if record["attempts"]>plan["max_retries"]:
                        status["status"]="failed";status["error"]=f"Training failed: {log.relative_to(ROOT)}"
                        atomic_json(args.state,status);return TERMINAL_FAILURE
                    continue
                record["attempts"]=0
                time.sleep(2.1)
                checkpoint=latest_checkpoint(directory,prefix)
                if checkpoint is None:raise RuntimeError("Trainer exited without a valid checkpoint")
            record["completed_iterations"]=int(re.search(r"model_(\d+)",checkpoint.name)[1])+1
            publish(checkpoint,args.models/name)
            status["current"]["phase"]="validation"
            status["current"]["completed_iterations"]=record["completed_iterations"]
            report=args.state.parent/f"{name}_{record['completed_iterations']:06d}_validation.json"
            evaluate=[str(ROOT/"run_isaac.sh"),"scripts/reinforcement_learning/rsl_rl/evaluate_native.py",
                      "--skill",name,"--checkpoint",str(checkpoint),"--num_envs",str(plan["evaluation_envs"]),
                      "--steps",str(plan["evaluation_steps"]),"--out",str(report)]
            code=run_child(evaluate,log.with_name(log.stem+"_validation.log"),status,args.state)
            if STOP:break
            if code==0 and report.exists():
                result=json.loads(report.read_text());record["evaluations"].append(str(report.relative_to(ROOT)))
                record["last_evaluated_iteration"]=record["completed_iterations"]
                record["last_validation_pass"]=result["pass_validation"]
                if result["pass_validation"] and result["score"]>record.get("best_score",-1.e9):
                    publish(checkpoint,args.models/name,True,str(report.relative_to(ROOT)))
                    record["best_score"]=result["score"];record["validated"]=True
            else:raise RuntimeError(f"Evaluation process failed: {log.with_name(log.stem+'_validation.log')}")
            atomic_json(args.state,status)
        record["budget_completed"]=record["completed_iterations"]>=stage["iterations"]
        if STOP:break
    status["status"]="stopped" if STOP else "completed" if all(s.get("validated") for s in status["stages"].values()) else "completed_with_unvalidated_skills"
    status["updated_at"]=datetime.now(timezone.utc).isoformat();atomic_json(args.state,status)
    return 0


if __name__=="__main__":
    try:code=main()
    except Exception as error:
        if STATE_PATH and STATE_PATH.exists():
            status=json.loads(STATE_PATH.read_text());status["status"]="failed";status["error"]=str(error)
            atomic_json(STATE_PATH,status)
        import traceback
        traceback.print_exc()
        code=TERMINAL_FAILURE
    raise SystemExit(code)
