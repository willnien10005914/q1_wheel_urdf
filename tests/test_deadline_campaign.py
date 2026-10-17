"""Real isolated Git delivery, with simulator execution replaced by test reports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('campaign',Path(__file__).resolve().parents[1]/'tools/deadline_campaign.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


class DeadlineDeliveryTest(unittest.TestCase):
    def test_failed_motion_is_archived_without_promotion_and_user_index_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'repo';root.mkdir();remote=base/'remote.git'
            def git(*args):return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout
            subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
            git('init','-b','main');git('config','user.email','test@example.invalid');git('config','user.name','Test')
            (root/'initial').write_text('base');git('add','initial');git('commit','-m','base');git('remote','add','origin',str(remote));git('push','origin','main')
            # User has unrelated staged work; the delivery must not touch it.
            (root/'user-edit').write_text('do not publish');git('add','user-edit')
            config=root/'config.json';config.write_text(json.dumps({'delivery_deadline':'2099-01-01T00:00:00+00:00',
                  'stages':[{'skill':'backflip'}]}))
            models=root/'checkpoints/native_manager';(models/'backflip').mkdir(parents=True)
            (models/'backflip/latest.pt').write_bytes(b'TEST ONLY: simulator is stubbed')
            (models/'backflip/latest.json').write_text(json.dumps({'validated':False}))
            state=root/'outputs/state/status.json';state.parent.mkdir(parents=True);state.write_text('{}')
            result=root/'docs/training/deadline_20261019/results'
            def fake_check(command,log,timeout):
                log.parent.mkdir(parents=True,exist_ok=True);log.write_text('TEST simulator stub\n')
                if '--out' in command:
                    dest=Path(command[command.index('--out')+1])
                    if dest.suffix=='.json':dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('{"pass_validation":false}')
                return 0
            # Skip actual browser by exhausting its optional time allowance only.
            real_time=time.time
            with patch.multiple(campaign,ROOT=root,CONFIG=config,STATE=state,RESULT=result,MODELS=models), \
                 patch.object(campaign,'run_checked',side_effect=fake_check), \
                 patch.object(campaign.time,'time',return_value=4070908800.-1800.-100.):
                self.assertEqual(campaign.finalize(),0)
            self.assertEqual(git('diff','--cached','--name-only').strip(),'user-edit')
            self.assertFalse((models/'backflip/model.pt').exists())
            delivery=json.loads((root/'outputs/native_deadline_20261019/final_review/delivery.json').read_text())
            self.assertEqual(delivery['status'],'pushed')
            paths=git('ls-tree','-r','--name-only',delivery['commit'])
            self.assertNotIn('user-edit',paths)
            self.assertIn('checkpoints/native_deadline_20261019/backflip/latest.pt',paths)
            summary=json.loads(git('show',delivery['commit']+':docs/training/deadline_20261019/results/summary.json'))
            self.assertFalse(summary['all_scheduled_motion_validations_passed'])
            self.assertFalse(summary['skills']['backflip']['pass_validation'])

    def test_missing_command_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'ROOT',Path(tmp)):
            log=Path(tmp)/'missing.log'
            self.assertEqual(campaign.run_checked(['/definitely/missing/q1-command'],log,1),127)
            self.assertTrue(log.read_text())

if __name__=='__main__':unittest.main()
