"""Verify every saved final prediction against reported per-task macro metrics."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
s=json.loads((root/'summary.json').read_text());history=[json.loads(x) for x in (root/'history.jsonl').read_text().splitlines()]
checked=0
for name,trial in s['trials'].items():
 rows=[r for r in history if r['trial']==name]
 assert [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
 with np.load(root/name/'last_predictions.npz') as d:
  assert len(set(d['ids']))==len(d['ids'])
  for task in ['cfp','oct']:
   prediction=d[task].argmax(1);y=d['y'];m=trial['fixed_last']['tasks'][task]
   assert np.isfinite(d[task]).all() and np.allclose(d[task].sum(1),1)
   for key,fn in [('macro_f1',f1_score),('macro_precision',precision_score),('macro_recall',recall_score)]:
    assert abs(fn(y,prediction,labels=[0,1],average='macro',zero_division=0)-m[key])<1e-12
   assert confusion_matrix(y,prediction,labels=[0,1]).tolist()==m['confusion_matrix'];checked+=1
report={'trials':len(s['trials']),'epochs':len(history),'prediction_task_checks':checked,'macro_metrics_and_confusion_matrices_match':True,'test_used':s['test_used']}
(root/'completion_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
