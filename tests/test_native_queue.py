"""Queue guards for partial saves, validation promotion and interrupted evaluation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import zipfile

spec=importlib.util.spec_from_file_location("native_queue",Path(__file__).resolve().parents[1]/"tools/native_training_queue.py")
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def checkpoint(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("archive/data.pkl",b"\x80\x04N.")
        archive.writestr("archive/data/0",b"x"*2048)
    stamp=time.time()-10;os.utime(path,(stamp,stamp))


class NativeQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(queue,"ROOT",self.root);self.patch.start();queue.CHECKPOINT_CACHE.clear();queue.STOP=False

    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def test_newest_complete_numeric_checkpoint(self):
        for n in (2,10):checkpoint(self.root/f"run/model_{n}.pt")
        partial=self.root/"run/model_11.pt";partial.write_bytes(b"incomplete"*300)
        stamp=time.time()-10;os.utime(partial,(stamp,stamp))
        self.assertEqual(queue.latest_checkpoint(self.root,"run").name,"model_10.pt")

    def test_fresh_save_is_not_published(self):
        checkpoint(self.root/"run/model_2.pt");checkpoint(self.root/"run/model_10.pt")
        os.utime(self.root/"run/model_10.pt",None)
        self.assertEqual(queue.latest_checkpoint(self.root,"run").name,"model_2.pt")

    def test_preview_and_validated_model_are_separate(self):
        source=self.root/"run/model_2.pt";checkpoint(source)
        queue.publish(source,self.root/"models")
        self.assertTrue((self.root/"models/latest.pt").exists())
        self.assertFalse((self.root/"models/model.pt").exists())
        queue.publish(source,self.root/"models",True,"evaluation.json")
        m=json.loads((self.root/"models/validated.json").read_text())
        self.assertTrue(m["validated"])
        self.assertEqual(m["sha256"],hashlib.sha256(source.read_bytes()).hexdigest())

    def test_resume_finishes_pending_evaluation_without_retraining(self):
        plan=dict(contract=queue.CONTRACT,num_envs=4,chunk_iterations=3,seed=42,evaluation_envs=2,evaluation_steps=4,
                  max_retries=0,stages=[dict(skill="stand",task="Isaac-Q1-Native-Stand-v0",iterations=3)])
        config=self.root/"plan.json";config.write_text(json.dumps(plan));state=self.root/"state/status.json"
        argv=["queue","--config",str(config),"--state",str(state),"--models",str(self.root/"models")]
        calls=[];first=True
        def child(command,log,status,state_path,stage=None,prefix=None,models=None):
            nonlocal first
            queue.atomic_json(state_path,status)
            if stage:
                calls.append("train");checkpoint(self.root/f"logs/rsl_rl/q1_native_stand/date_{prefix}0/model_2.pt")
            else:
                calls.append("evaluate")
                if first:first=False;raise RuntimeError("simulated interruption during evaluation")
                report=Path(command[command.index("--out")+1]);report.write_text(json.dumps(dict(pass_validation=True,score=1.)))
            return 0
        with patch.object(sys,"argv",argv),patch.object(queue,"source_manifest",return_value={}), \
             patch.object(queue,"run_child",side_effect=child),patch.object(queue.time,"sleep"), \
             patch.object(queue.signal,"signal"),patch.object(queue.fcntl,"flock"):
            with self.assertRaisesRegex(RuntimeError,"simulated interruption"):queue.main()
            self.assertEqual(queue.main(),0)
        self.assertEqual(calls,["train","evaluate","evaluate"])
        result=json.loads(state.read_text())
        self.assertEqual(result["stages"]["stand"]["last_evaluated_iteration"],3)
        self.assertTrue(result["stages"]["stand"]["validated"])

    def test_training_chunks_advance_saved_iteration(self):
        plan=dict(contract=queue.CONTRACT,num_envs=4,chunk_iterations=1,seed=42,evaluation_envs=2,evaluation_steps=4,
                  max_retries=0,stages=[dict(skill="stand",task="Isaac-Q1-Native-Stand-v0",iterations=2)])
        config=self.root/"plan.json";config.write_text(json.dumps(plan));state=self.root/"state/status.json"
        argv=["queue","--config",str(config),"--state",str(state),"--models",str(self.root/"models")]
        training=[]
        def child(command,log,status,state_path,stage=None,prefix=None,models=None):
            queue.atomic_json(state_path,status)
            if stage:
                iteration=0
                if "--resume" in command:
                    self.assertIn("--resume_next_iteration",command)
                    iteration=int(Path(command[command.index("--checkpoint")+1]).stem.split("_")[1])+1
                training.append(iteration)
                checkpoint(self.root/f"logs/rsl_rl/q1_native_stand/date_{prefix}{iteration}/model_{iteration}.pt")
            else:
                Path(command[command.index("--out")+1]).write_text(json.dumps(dict(pass_validation=True,score=1.)))
            return 0
        with patch.object(sys,"argv",argv),patch.object(queue,"source_manifest",return_value={}), \
             patch.object(queue,"run_child",side_effect=child),patch.object(queue.time,"sleep"), \
             patch.object(queue.signal,"signal"),patch.object(queue.fcntl,"flock"):
            self.assertEqual(queue.main(),0)
        self.assertEqual(training,[0,1])
        self.assertEqual(json.loads(state.read_text())["stages"]["stand"]["completed_iterations"],2)

    def test_exhausted_retries_stay_terminal_after_supervisor_restart(self):
        plan=dict(contract=queue.CONTRACT,num_envs=4,chunk_iterations=1,seed=42,evaluation_envs=2,evaluation_steps=4,
                  max_retries=2,stages=[dict(skill="stand",task="Isaac-Q1-Native-Stand-v0",iterations=2)])
        config=self.root/"plan.json";config.write_text(json.dumps(plan));state=self.root/"state/status.json"
        argv=["queue","--config",str(config),"--state",str(state),"--models",str(self.root/"models")]
        def failed_child(command,log,*args):
            log.parent.mkdir(parents=True,exist_ok=True);log.write_text("startup failed")
            return 1
        with patch.object(sys,"argv",argv),patch.object(queue,"source_manifest",return_value={}), \
             patch.object(queue,"run_child",side_effect=failed_child) as child, \
             patch.object(queue.signal,"signal"),patch.object(queue.fcntl,"flock"):
            self.assertEqual(queue.main(),queue.TERMINAL_FAILURE)
            self.assertEqual(child.call_count,3)
            self.assertEqual(queue.main(),queue.TERMINAL_FAILURE)
            self.assertEqual(child.call_count,3)
        self.assertEqual(json.loads(state.read_text())["status"],"failed")

    def test_stalled_child_is_killed_and_returns_retryable_timeout(self):
        child=MagicMock();child.pid=12345;child.poll.return_value=None
        child.wait.side_effect=[queue.subprocess.TimeoutExpired('sim',30),0]
        state={"current":{}}
        with patch.object(queue.subprocess,"Popen",return_value=child), \
             patch.object(queue.time,"monotonic",side_effect=[0.,601.]), \
             patch.object(queue.os,"killpg") as kill,patch.object(queue,"STALL_TIMEOUT",600.):
            code=queue.run_child(['sim'],self.root/'child.log',state,self.root/'status.json')
        self.assertEqual(code,124)
        self.assertEqual([c.args for c in kill.call_args_list],[(12345,queue.signal.SIGTERM),(12345,queue.signal.SIGKILL)])
        self.assertIn('no log output',state['current']['watchdog_error'])
        self.assertIsNone(queue.CHILD)


if __name__=="__main__":unittest.main()
