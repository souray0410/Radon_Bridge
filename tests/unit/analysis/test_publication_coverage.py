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
        a=check(self.root);self.assertEqual(a['unique_accepted_executions'],45)  # Centered adds two new executions; two rows strictly reuse core.
        self.assertEqual(a['accepted_packages'],6)
        self.assertEqual(a['indexed_packages'],11)
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

    def test_centered_locked_identity_rejected(self):
        self.mutate('coverage.json',lambda d:next(p for p in d['packages'] if p['id']=='ws02_centered_basis')['comparisons'].pop())
        with self.assertRaisesRegex(ValueError,'centered WS02'):check(self.root)

    def test_centered_fixed_identity_rejected(self):
        self.mutate('coverage.json',lambda d:next(p for p in d['packages'] if p['id']=='ws02_centered_basis')['fixed'].update(r=64))
        with self.assertRaisesRegex(ValueError,'centered WS02'):check(self.root)

    def test_centered_acceptance_requires_global_reuse_registration(self):
        def remove_centered_reuse(d):
            d['reuse']=[row for row in d['reuse'] if row['target_package']!='ws02_centered_basis']
        self.mutate('coverage.json',remove_centered_reuse)
        with self.assertRaisesRegex(ValueError,'global reuse'):check(self.root)

    def test_centered_acceptance_review_identity_is_locked(self):
        self.mutate('coverage.json',lambda d:next(p for p in d['packages'] if p['id']=='ws02_centered_basis')['acceptance_review'].update(source_files_exact=90))
        with self.assertRaisesRegex(ValueError,'centered WS02 review identity'):check(self.root)

    def test_centered_strict_reuse_identity_rejected_if_prediction_changes(self):
        self.mutate('centered_basis/current.json',lambda d:next(r for r in d['results'] if r['id']=='uncentered_radon').update(prediction_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'Reused prediction'):check(self.root)

    def test_duplicate_reuse_target_rejected(self):
        def duplicate(d):
            row=dict(d['reuse'][0]);row['source_id']='another_source';d['reuse'].append(row)
        self.mutate('coverage.json',duplicate)
        with self.assertRaisesRegex(ValueError,'Duplicate reuse target'):check(self.root)

    def test_missing_page_rejected(self):
        (self.root/'factorized/README.md').unlink()
        with self.assertRaisesRegex(ValueError,'Missing coverage'):check(self.root)


if __name__=='__main__':unittest.main()
