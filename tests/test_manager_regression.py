"""CPU-only guards against false passes in recorded behavior comparisons."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_runtime import manager_registrations
from tools.manager_regression import compare_rollouts


class ManagerRegressionTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = {"observation": .1, "action": .05, "reward": .1, "height": .02,
                           "position": .1, "gravity": .05, "command": 1e-6, "termination_count": 0}
        self.baseline = {"metadata": {"task": "manager-test", "seed": 4107, "steps": 1,
                                      "num_envs": 1, "step_dt": .02,
                                      "schema": {"action_dim": 2}, "fixed_commands": {},
                                      "checkpoint_sha256": "stable"},
                         "trace": [{"observation": [[0., 0., 1.]], "action": [[0., 0.]],
                                    "reward": [1.], "height": [.7], "position": [[0., 0., .7]],
                                    "gravity": [[0., 0., -1.]], "command": {"velocity": [[0., 0., 0.]]},
                                    "done": [False], "terminated": [False], "truncated": [False]}]}

    def compare(self, candidate, **kwargs):
        return compare_rollouts(self.baseline, candidate, self.thresholds, **kwargs)

    def test_identical_capture_passes(self):
        self.assertTrue(self.compare(copy.deepcopy(self.baseline))["pass"])

    def test_observation_contract_change_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["metadata"]["schema"]["action_dim"] = 3
        self.assertFalse(self.compare(candidate)["pass"])

    def test_checkpoint_change_requires_explicit_flag(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["metadata"]["checkpoint_sha256"] = "trained"
        self.assertFalse(self.compare(candidate)["pass"])
        self.assertTrue(self.compare(candidate, allow_checkpoint_change=True)["pass"])

    def test_action_drift_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["trace"][0]["action"] = [[.2, .2]]
        self.assertFalse(self.compare(candidate)["pass"])

    def test_nonfinite_values_are_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["trace"][0]["height"] = [float("nan")]
        self.assertFalse(self.compare(candidate)["pass"])

    def test_extra_termination_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["trace"][0]["done"] = [True]
        candidate["trace"][0]["terminated"] = [True]
        self.assertFalse(self.compare(candidate)["pass"])

    def test_incomplete_capture_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["trace"] = []
        self.assertFalse(self.compare(candidate)["pass"])

    def test_all_task_declarations_use_manager_env(self):
        rows = manager_registrations(Path(__file__).resolve().parents[1])
        self.assertTrue(any(row["task"] == "Isaac-Q1-Skate-v0" for row in rows))
        self.assertTrue(any(row["task"] == "Isaac-Q1-RecoveryV3-BootKneel-v0" for row in rows))
        self.assertTrue(all(row["entry_point"] == "isaaclab.envs:ManagerBasedRLEnv" for row in rows))


if __name__ == "__main__":
    unittest.main()
