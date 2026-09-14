import hashlib,json,time
from pathlib import Path
import numpy as np
root=Path('/data/mengh/Radon_Bridge/runs/2026_09_14_19_43_11')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def f1(y,p):
 pred=p.argmax(1);scores=[]
 for c in (0,1):
  tp=((y==c)&(pred==c)).sum();fp=((y!=c)&(pred==c)).sum();fn=((y==c)&(pred!=c)).sum()
  scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(scores))
q=json.loads((root/'queue.json').read_text());dep=json.loads((root/'dependencies_acceptance.json').read_text())
assert dep['passed'] and dep['test_used'] is False and sha(root/'queue.json')==dep['queue_sha256']
# Check small immutable source/config/parent files; raw training cache was accepted
# before execution and is not rescanned by this artifact reacceptance.
for section in ('config_files','references'):
 for p,h in dep[section].items():assert sha(p)==h,(section,p)
code=json.loads((root/'code_acceptance.json').read_text())
for p,h in code['files'].items():assert sha(p)==h,p
rows=[];all_ids=None;all_y=None
for c in q['cases']:
 p=root/'trials'/c['name'];a=json.loads((p/'accepted.json').read_text());cfg=json.loads(Path(c['config']).read_text())
 assert a['configuration']==cfg and a['test_used'] is False and a['state']=='complete'
 assert a['identity']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
 assert a['converged_by_policy'] and a['stop_reason']=='validation_plateau'
 required={'best.pt','resume.pt','history.json','initial_acceptance.json','selected_predictions.npz'}
 assert required <= set(a['files'])
 for n,h in a['files'].items():assert sha(p/n)==h,(c['name'],n)
 with np.load(p/'initial_predictions.npz',allow_pickle=False) as z:
  ids=z['ids'].copy();y=z['y'].copy();initial=(f1(y,z['cfp'])+f1(y,z['oct']))/2
  for modality,ref in cfg['parents'].items():
   with np.load(Path(ref['path']).parent/'selected_predictions.npz',allow_pickle=False) as par:
    assert np.array_equal(ids,par['ids']) and np.array_equal(y,par['y'])
    np.testing.assert_allclose(z[modality],par[modality],rtol=1e-5,atol=1e-6)
 assert len(ids)==296 and len(set(ids.tolist()))==296
 if all_ids is None:all_ids=ids;all_y=y
 assert np.array_equal(ids,all_ids) and np.array_equal(y,all_y)
 best=anchor=initial;best_epoch=0;bad=0;hist=json.loads((p/'history.json').read_text())
 assert len(hist)==a['epochs_ran'] and 8<=len(hist)<=60
 for epoch,row in enumerate(hist,1):
  assert row['epoch']==epoch
  score=row['metrics']['mean_task_macro_f1'];assert np.isfinite(score)
  improved=score>best
  if improved:best=score;best_epoch=epoch
  if score>anchor+.001:anchor=score;bad=0
  else:bad+=1
  expected={'improved':improved,'reduce_lr':bad>0 and bad%3==0,'plateau':epoch>=8 and bad>=6}
  assert row['flags']==expected,(c['name'],epoch)
  assert not expected['plateau'] or epoch==len(hist)
 assert bad>=6 and best_epoch==a['best_epoch']
 with np.load(p/'selected_predictions.npz',allow_pickle=False) as z:
  assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y'])
  values={m:f1(y,z[m]) for m in ('cfp','oct')}
  for m,v in values.items():assert abs(v-a['selected_validation']['tasks'][m]['macro_f1'])<1e-12
  assert abs(np.mean(list(values.values()))-best)<1e-12
 rows.append(dict(name=c['name'],seed=cfg['seed'],cfp_f1=values['cfp'],oct_f1=values['oct'],mean_f1=best,best_epoch=best_epoch,stop_epoch=len(hist),receipt_sha256=sha(p/'accepted.json')))
 print('accepted',c['name'],flush=True)
result=dict(checked_at=time.time(),scope='independent_artifact_hash_config_parent_prediction_plateau_and_metric_reacceptance',accepted=len(rows),expected=len(q['cases']),raw_cache_rescanned=False,new_model_inference=False,test_used=False,queue_sha256=sha(root/'queue.json'),script_sha256=sha(__file__),rows=rows)
out=root/'independent_acceptance_20260915';out.mkdir(exist_ok=False)
(out/'accepted.json').write_text(json.dumps(result,indent=2))
print('ACCEPTED',len(rows),'report',out)
