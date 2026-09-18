"""Regression checks for the real missing-supplement publication failure."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from radon_bridge.analysis.publication_coverage import check


class PublicationCoverageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'current'
        shutil.copytree(Path(__file__).resolve().parents[3]/'docs/reports/current', self.root)

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, file, func):
        p=self.root/file;d=json.loads(p.read_text());func(d);p.write_text(json.dumps(d))

    def test_current_and_idempotent(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        a=check(self.root);self.assertEqual(a['unique_accepted_executions'],43)  # Adds eight new grouped runs; G=1 is strict reuse.
        self.assertFalse(a['full_project_accepted']);self.assertEqual(a,check(self.root))
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_missing_negative_or_any_result_rejected(self):
        self.mutate('factorized/current.json',lambda d:d['results'].pop())
        with self.assertRaisesRegex(ValueError,'Missing'):check(self.root)

    def test_duplicate_rejected(self):
        self.mutate('factorized/current.json',lambda d:d['results'].append(d['results'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):check(self.root)

    def test_entire_supplement_omission_rejected(self):
        self.mutate('coverage.json',lambda d:d['packages'].pop(1))
        with self.assertRaisesRegex(ValueError,'Approved package'):check(self.root)

    def test_changed_reuse_rejected(self):
        self.mutate('small_cohort/current.json',lambda d:d['results'][0].update(prediction_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'Reused prediction'):check(self.root)

    def test_grouped_g1_reuse_rejected_if_identity_changes(self):
        self.mutate('grouped_linear/current.json',lambda d:d['results'][0].update(prediction_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'Reused prediction'):check(self.root)

    def test_incomplete_cannot_claim_accepted(self):
        self.mutate('small_cohort/current.json',lambda d:d.update(complete=False))
        with self.assertRaisesRegex(ValueError,'Unaccepted'):check(self.root)

    def test_mechanism_removal_rejected(self):
        self.mutate('coverage.json',lambda d:next(p for p in d['packages'] if p['id']=='ibex_mechanisms')['expected_arms'].pop())
        with self.assertRaisesRegex(ValueError,'mechanism'):check(self.root)

    def test_grouped_locked_identity_rejected(self):
        self.mutate('coverage.json',lambda d:next(p for p in d['packages'] if p['id']=='grouped_linear').update(groups=[1,2,4,8]))
        with self.assertRaisesRegex(ValueError,'grouped WS02'):check(self.root)

    def test_missing_page_rejected(self):
        (self.root/'factorized/README.md').unlink()
        with self.assertRaisesRegex(ValueError,'Missing coverage'):check(self.root)


if __name__=='__main__':unittest.main()
