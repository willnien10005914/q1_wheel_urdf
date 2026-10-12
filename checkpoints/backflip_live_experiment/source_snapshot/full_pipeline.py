"""Authorized 6144/20k run followed by independent evaluation and video upload."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
os.chdir(ROOT)
STATE=Path('v2_full_pipeline.json')
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,choices=[6144,12288],default=6144)
parser.add_argument('--resume')
args=parser.parse_args()
def save(**data):
    prior=json.loads(STATE.read_text()) if STATE.exists() else {}
    if prior.get('pid')!=os.getpid():prior={}
    STATE.write_text(json.dumps({**prior,'pid':os.getpid(),'num_envs':args.num_envs,'target_iterations':20000,'updated':time.strftime('%Y-%m-%d %H:%M:%S'),**data},indent=2))
def run(command,log):
    with Path(log).open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
def gate(suffix,num_envs,teacher_hash):
    matches=sorted(Path('logs/q1_backflip_v2').glob('*_'+suffix+'/completed.json'))
    if not matches:raise RuntimeError('Missing training gate: '+suffix)
    path=matches[-1];data=json.loads(path.read_text());manifest=json.loads((path.parent/'manifest.json').read_text())
    if not data['finite'] or data['num_envs']!=num_envs:raise RuntimeError('Failed numeric/capacity gate: '+suffix)
    if manifest['files'][manifest['args']['teacher']]!=teacher_hash:raise RuntimeError('Gate used a different teacher')
    for source in ['q1_backflip/env.py','q1_backflip_v2/env.py','q1_backflip_v2/compact_env.py','q1_backflip_v2/landing_env.py']:
        if manifest['files'][source]!=hashlib.sha256(Path(source).read_bytes()).hexdigest():raise RuntimeError('Changed gate environment: '+source)

try:
    teacher=Path('q1_backflip_v2/teacher_landing_balance.json')
    digest=hashlib.sha256(teacher.read_bytes()).hexdigest()
    gate('landing_balance_probe64_100',64,digest)
    gate('landing_balance_capacity6144_3',6144,digest)
    if args.num_envs==12288:gate('landing_balance_benchmark12288_30',12288,digest)
    evaluation=json.loads(Path('artifacts/v2/landing_balance_eval96/evaluation.json').read_text())
    if evaluation['teacher_sha256']!=digest or evaluation['successes']<1 or evaluation['early_hops']!=0:raise RuntimeError('No independently validated no-hop backflip landing')
    save(state='training',num_envs=args.num_envs,target_iterations=20000,teacher_sha256=digest,resume=args.resume)
    log=f'v2_landing{args.num_envs}_20k.log'
    command=['bash','q1_backflip_v2/launch.sh','--num_envs',str(args.num_envs),'--max_iterations','20000','--landing','kneel','--teacher',str(teacher),'--noise_std','0.05','--run_name',f'landing{args.num_envs}_20k','--save_interval','200']
    if args.resume:command+=['--resume',args.resume]
    run(command,log)
    directory=None
    with Path(log).open() as stream:
        for line in stream:
            if line.startswith('RUN_DIR '):directory=Path(line[len('RUN_DIR '):].split(' START_ITER ')[0].strip())
    if directory is None or not (directory/'completed.json').exists():raise RuntimeError('Final training completion missing')
    checkpoint=directory/'model_19999.pt'
    if not checkpoint.exists():raise RuntimeError('Final checkpoint missing')
    save(state='evaluating',run_dir=str(directory),checkpoint=str(checkpoint))
    out=Path('artifacts/v2/final20k')
    run(['bash','q1_backflip_v2/evaluate.sh','--checkpoint',str(checkpoint),'--out',str(out/'eval96')],'v2_final20k_eval96.log')
    run(['bash','q1_backflip_v2/evaluate.sh','--checkpoint',str(checkpoint),'--out',str(out/'video'),'--video','--steps','300'],'v2_final20k_video.log')
    result=json.loads((out/'eval96/evaluation.json').read_text())
    save(state='uploading',run_dir=str(directory),successes=result['successes'],evaluations=result['num_completed_episodes'])
    run(['python3','/home/will/projects/q1_wheel_urdf/tools/upload_gofile.py',str(out/'video/q1_v2_actual.mp4'),str(out/'video/upload.gofile.json')],'v2_final20k_upload.log')
    upload=json.loads((out/'video/upload.gofile.json').read_text())
    save(state='completed',run_dir=str(directory),checkpoint=str(checkpoint),successes=result['successes'],evaluations=result['num_completed_episodes'],early_hops=result['early_hops'],video_url=upload['data']['downloadPage'])
except BaseException as error:
    save(state='failed',error=str(error))
    raise
