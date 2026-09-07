"""Negative gates and independent numerical references for unified evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.metrics import f1_score
from tools.audit_current_model_lineage import dataset_splits
from radonbridge.unified_evaluation import verify_lock, validate_prediction
from scripts.report_unified_test import bootstrap_f1, intervals


class UnifiedChecks(unittest.TestCase):
    def test_forbid_test_and_dynamic_training_sources(self):
        good="a=PairedDataset(root,'train'); b=PairedDataset(root,'validation')"
        self.assertEqual(len(dataset_splits(good)),2)
        for extra in ("; c=PairedDataset(root,'test')","; c=PairedDataset(root,split)"):
            with self.assertRaises(ValueError):dataset_splits(good+extra)
    def test_test_requires_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'candidate_lock.json').write_text(json.dumps({'files':{}}))
            verify_lock(p,'validation')
            with self.assertRaises(FileNotFoundError):verify_lock(p,'test')
            with self.assertRaises(ValueError):verify_lock(p,'train')
    def test_changed_locked_files_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'jobs.json').write_text('[]');(p/'candidate_lock.json').write_text(json.dumps({'files':{'jobs.json':'wrong'}}))
            with self.assertRaises(ValueError):verify_lock(p,'validation')
    def test_bootstrap_matches_sklearn(self):
        y=np.array([0,0,0,1,1,1]);pred=np.array([[[0,0,1,0,1,1],[1,0,0,1,0,1]],[[0,0,0,0,0,0],[1,1,1,1,1,1]]])
        indices=np.array([[0,1,2,3,4,5],[0,0,1,1,2,2],[5,5,5,5,5,5],[0,1,3,4,5,5]])
        point,draw=bootstrap_f1(y,pred,indices)
        for j,p in enumerate(pred.reshape(-1,6)):
            self.assertAlmostEqual(point[j],100*f1_score(y,p,average='macro',labels=[0,1],zero_division=0))
            for i,ix in enumerate(indices):self.assertAlmostEqual(draw[i,j],100*f1_score(y[ix],p[ix],average='macro',labels=[0,1],zero_division=0))
    def test_zero_variance_and_unestimable(self):
        defs=[dict(contrast_id=str(i),primary=True,estimable=i!=2,aliases=[dict(family='f',primary=True)]) for i in range(3)]
        draws=np.stack([np.linspace(-2,2,100),np.ones(100),np.zeros(100)],axis=1)
        rows,c=intervals(defs,np.array([0.,1.,0.]),draws)
        self.assertIsNotNone(rows[0]['global_simultaneous_ci95_pp'])
        self.assertEqual(rows[1]['classification'],'undefined_zero_variance')
        self.assertEqual(rows[2]['classification'],'not_estimable')
    def test_identity_and_probability_checks(self):
        class Data:
            rows=[dict(order='a',label_id=0),dict(order='b',label_id=1)]
            def __len__(self):return 2
        a=dict(ids=np.array(['a','b']),y=np.array([0,1]),cfp=np.array([[.7,.3],[.2,.8]]),oct=np.array([[.8,.2],[.1,.9]]))
        validate_prediction(a,Data());a['ids']=a['ids'][::-1]
        with self.assertRaises(ValueError):validate_prediction(a,Data())


if __name__=='__main__':unittest.main()
