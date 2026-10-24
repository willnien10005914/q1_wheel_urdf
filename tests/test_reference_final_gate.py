"""Independent final validation must gate published Web policies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import reference_campaign as campaign


class FinalValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.weight=self.root/'model.pt'
        self.weight.write_bytes(b'TEST weights only')
        self.state=dict(reference_sha256='reference',urdf_sha256='urdf')
        self.stage=dict(skill='kneel',policy_contract=campaign.POLICY_CONTRACT)
        self.meta=dict(contract=campaign.POLICY_CONTRACT,skill='kneel',sha256=campaign.sha256(self.weight),
                       reference_sha256='reference',urdf_sha256='urdf',validated=True,validation='earlier.json')
        self.metadata=self.root/'validated.json';self.metadata.write_text(json.dumps(self.meta))
        self.report=self.root/'final.json'
        self.result=dict(contract=campaign.POLICY_CONTRACT,skill='kneel',checkpoint_sha256=campaign.sha256(self.weight),
                         reference_sha256='reference',urdf_sha256='urdf',first_trial_only=True,
                         seed=10424,trials=64,nonfinite=0,pass_validation=True)

    def gate(self,**changes):
        self.report.write_text(json.dumps(dict(self.result,**changes)))
        with patch.object(campaign,'ROOT',self.root):
            return campaign.final_validation_gate(self.weight,self.report,self.stage,self.state,0)

    def test_failed_final_revokes_web_and_preserves_checkpoint(self):
        original=self.weight.read_bytes();decision=self.gate(pass_validation=False)
        self.assertFalse(decision['pass_validation']);self.assertFalse(decision['web_enabled'])
        metadata=json.loads(self.metadata.read_text())
        self.assertFalse(metadata['validated']);self.assertTrue(metadata['training_validated'])
        self.assertEqual(self.weight.read_bytes(),original)

    def test_same_shape_wrong_reference_cannot_pass_final(self):
        self.assertFalse(self.gate(reference_sha256='wrong-reference')['web_enabled'])

    def test_final_checks_cannot_use_training_seed_or_reset_trials(self):
        self.assertFalse(self.gate(seed=42)['web_enabled'])
        self.assertFalse(self.gate(first_trial_only=False)['web_enabled'])

    def test_valid_final_replaces_earlier_validation_for_sealing(self):
        self.assertTrue(self.gate()['web_enabled'])
        self.assertEqual(json.loads(self.metadata.read_text())['validation'],'final.json')

    def test_formal_metadata_without_training_acceptance_is_not_promoted(self):
        self.meta['validated']=False;self.metadata.write_text(json.dumps(self.meta))
        self.assertFalse(self.gate()['web_enabled'])

    def test_invalid_final_json_revokes_web(self):
        self.report.write_text('{incomplete')
        with patch.object(campaign,'ROOT',self.root):
            decision=campaign.final_validation_gate(self.weight,self.report,self.stage,self.state,0)
        self.assertFalse(decision['web_enabled'])


if __name__=='__main__':unittest.main()
