"""Resource handover tests; no real GPU allocation or upstream mutations."""
import sys,tempfile,json,fcntl,subprocess
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import reference_campaign_guard as g
import reference_campaign as c

class CampaignTest(unittest.TestCase):
 def test_actual_plan_has_6144_and_10000_per_five_skills(self):
  cfg=c.load_json(ROOT/'config/reference_campaign_20261024.json');plan=c.queue_plan(cfg)
  self.assertEqual(plan['num_envs'],6144);self.assertEqual(len(plan['stages']),5)
  self.assertTrue(all(s['iterations']==10000 for s in plan['stages']))
  smoke=c.queue_plan(cfg,smoke=True);self.assertEqual(smoke['num_envs'],64);self.assertTrue(all(s['iterations']==3 for s in smoke['stages']))
 def test_reference_gate_checks_real_derived_files(self):
  cfg=c.load_json(ROOT/'config/reference_campaign_20261024.json');self.assertTrue(c.reference_gate(cfg)['ready'])
 def test_quiet_window_restarts_after_restart_gap(self):
  q=g.QuietWindow(60);self.assertFalse(q.observe(True,0));self.assertFalse(q.observe(False,59));self.assertFalse(q.observe(True,60));self.assertFalse(q.observe(True,119));self.assertTrue(q.observe(True,120))
 def test_interrupted_upstream_is_not_completion(self):
  self.assertNotIn('interrupted',g.TERMINAL_UPSTREAM);self.assertNotIn('runtime_error',g.TERMINAL_UPSTREAM)
  self.assertIn('budget_exhausted',g.TERMINAL_UPSTREAM)
 def test_handover_respects_upstream_real_flock(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'upstream.lock';path.write_text('preserved')
   with path.open('r') as owner:
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=g.HandoverLocks(path,Path(tmp)/'gpu.lock');self.assertFalse(lock.acquire())
   self.assertTrue(lock.acquire());lock.release();self.assertEqual(path.read_text(),'preserved')
 def test_gpu_compute_consumer_blocks_even_with_memory_available(self):
  def runner(cmd):
   return '0, GPU-X, RTX 4090, 16376, 7000, 80' if '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu' in cmd else 'GPU-X, 1234, python, 6900'
  snap=g.gpu_snapshot(runner=runner);self.assertEqual(len(snap['compute_apps']),1);self.assertFalse(snap['parallel_training_enabled'])
 def test_validated_metadata_is_not_sealed_without_matching_full_report(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'models/standup';folder.mkdir(parents=True);weight=folder/'model.pt';weight.write_bytes(b'TEST ONLY')
   digest=c.sha256(weight);meta={'contract':c.POLICY_CONTRACT,'validated':True,'sha256':digest,'validation':'missing.json'};(folder/'validated.json').write_text(json.dumps(meta))
   with patch.object(c,'ROOT',root):self.assertEqual(c.seal_models(root/'models',[{'skill':'standup','policy_contract':c.POLICY_CONTRACT}],'ref','urdf'),[])
   self.assertNotIn('reference_sha256',json.loads((folder/'validated.json').read_text()))
if __name__=='__main__':unittest.main()

class GuardRestartTest(unittest.TestCase):
 def test_irrelevant_process_does_not_require_cwd_permission(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'1389';p.mkdir();(p/'cmdline').write_bytes(b'codex\0daemon\0')
   with patch.object(g,'process_identity',side_effect=PermissionError('nondumpable daemon')):
    self.assertEqual(g.scan_upstream_processes('/project/q1_leg',root),[])
 def test_latest_same_project_run_replaces_stale_interrupted_run(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/'runs/new';run.mkdir(parents=True);proc=root/'proc';proc.mkdir()
   status=root/'latest.json';value={'robot':'q1_leg_upper_body_v6','run':str(run),'state':'budget_exhausted'}
   status.write_text(json.dumps(value));(run/'supervisor_state.json').write_text(json.dumps(value))
   def runner(cmd):
    if cmd[0]=='systemctl':return 'inactive'
    if '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu' in cmd:return '0, GPU-X, RTX, 16376, 100, 0'
    return ''
   config={'repo':str(root),'services':['q1-leg-upper-training.service'],'status_files':[{'path':'stale'}],
    'follow_latest_run':{'status_path':str(status),'run_root':str(root/'runs'),'expected_robot':'q1_leg_upper_body_v6'}}
   snapshot=g.upstream_snapshot(config,runner=runner,proc_root=proc)
   self.assertTrue(snapshot['ready']);self.assertEqual(snapshot['tracked_run'],str(run))
