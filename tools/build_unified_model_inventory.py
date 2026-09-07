"""Read-only, pre-test candidate inventory. It cannot authorize or execute test inference."""
import argparse,collections,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from radonbridge.artifacts import SOURCE,ARCHIVE,STUDY,resolve
from scripts.geometry_evidence import read,write
from radonbridge.geometry_study import semantic

SOURCES=[('current_direct_and_host',SOURCE/'runs'/STUDY/'branch_only/manifest.json',378),
 ('current_frozen',SOURCE/'runs'/STUDY/'dependency_supplement/manifest.json',6),
 ('history_final213',ARCHIVE/'history/runs/2026_09_05_22_42_53/manifest.json',213),
 ('s_axis18',ARCHIVE/'history/runs/2026_09_06_10_12_59/manifest.json',18),
 ('completion72',SOURCE/'runs'/STUDY/'pretest_completion_v2/manifest.json',72),
 ('multidepth96',SOURCE/'runs'/STUDY/'multidepth96/manifest.json',96)]

def canonical(x):
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x:return {k:canonical(v) for k,v in x.items() if k!='path'}
  return {k:canonical(v) for k,v in x.items()}
 if isinstance(x,list):return [canonical(v) for v in x]
 return x

def digest_json(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main(out):
 assert not out.exists(),'Use a unique snapshot directory; never overwrite an inventory'
 out.mkdir(parents=True);start=time.time();cache={};models={};refs=[];issues=[];sources=[];parents={};anchor=None
 def sha(path):
  p=resolve(path);stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
  if key not in cache:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
   assert p.stat().st_size==stat.st_size and p.stat().st_mtime_ns==stat.st_mtime_ns,'Artifact changed during read'
   cache[key]=h.hexdigest()
  return cache[key]
 def prediction(path,expected=None):
  nonlocal anchor
  actual=sha(path)
  if expected:assert actual==expected,'Prediction SHA mismatch'
  with np.load(resolve(path),allow_pickle=False) as z:
   assert len(z['ids'])==len(set(z['ids']))==len(z['y'])==296
   for k in ('cfp','oct'):
    assert z[k].shape==(296,2) and np.isfinite(z[k]).all() and np.allclose(z[k].sum(1),1,atol=1e-5)
   if anchor is None:anchor=(z['ids'].copy(),z['y'].copy())
   else:assert np.array_equal(anchor[0],z['ids']) and np.array_equal(anchor[1],z['y']),'Development order or labels mismatch'
  return actual
 def add(cfg,checkpoint,pred,summary,ref,expected=None,training_checkpoint=None):
  assert not cfg.get('task_fusion'),'Withdrawn learnable end-fusion protocol'
  sh=sha(checkpoint)
  if expected:assert sh==expected,'Checkpoint SHA mismatch'
  inference=dict(factory='PilotGraph',seed=cfg['seed'],bridges=canonical(cfg['bridges']),task_fusion=None,
      evaluation_state='eval_all_communications_enabled',preprocessing='CFP224_fullspan32_OCT96_v1')
  key=digest_json(dict(checkpoint_sha256=sh,inference=inference))
  ps=prediction(pred)
  record=models.setdefault(key,dict(model_view_id=key,checkpoint_path=str(resolve(checkpoint)),checkpoint_sha256=sh,
    configuration=cfg,inference_identity=inference,development_prediction_path=str(resolve(pred)),development_prediction_sha256=ps,
    reference_ids=[],strict_model_load_verified=False,development_forward_replay_verified=False,test_inference_permitted=False))
  assert record['development_prediction_sha256']==ps,'Identical model/inference mapped to conflicting predictions'
  record['reference_ids'].append(ref['reference_id']);refs.append(dict(ref,model_view_id=key,
    summary_path=str(resolve(summary)),summary_sha256=sha(summary),training_checkpoint_sha256=training_checkpoint or sh))
 for group,path,count in SOURCES:
  manifest=read(path);assert len(manifest['rows'])==count,(group,len(manifest['rows']))
  sources.append(dict(group=group,path=str(path),sha256=sha(path),training_references=count))
  for row in manifest['rows']:
   ref=dict(reference_id=group+':'+row['id'],source_group=group,source_row_id=row['id'],seed=row.get('seed'),arm=row.get('arm'),
      structure=row.get('structure'),rho=row.get('rho'),category=row.get('category'),regime='single_width')
   try:
    if row.get('state') not in (None,'accepted'):raise ValueError('Row not accepted: '+str(row.get('state')))
    directory=resolve(row['directory']);cfg=read(directory/'configuration.json');summary=read(directory/'summary.json')
    assert summary['state']=='complete' and summary['converged_by_policy'] and summary['stop_reason']=='validation_plateau' and summary['test_used'] is False
    assert semantic(cfg)==semantic(row['configuration'])==semantic(summary['configuration'])
    for name,expected in row.get('accepted_hashes',{}).items():assert sha(directory/name)==expected,('Artifact SHA mismatch',name)
    for branch,value in cfg.get('parent_checkpoints',{}).items():
     assert sha(value['path'])==value['sha256'],'Parent SHA mismatch'
     parents.setdefault(cfg['seed'],{})[branch]=value
    for bridge in cfg['bridges']:
     for value in bridge.get('basis_files',{}).values():assert sha(value['path'])==value['sha256'],'Basis SHA mismatch'
    nested=any(b.get('nested_rhos') for b in cfg['bridges'])
    if nested:
     exports=read(directory/'width_exports.json');assert len(exports)==3
     train_sha=sha(directory/'selected.pt')
     for exp in exports:
      q=resolve(exp['directory']);small=read(q/'configuration.json');r=dict(ref,reference_id=ref['reference_id']+':rho='+str(exp['rho']),rho=exp['rho'],regime='joint_nested_width_view')
      assert small['width_training_provenance']['regime']=='joint_nested'
      add(small,q/'selected.pt',q/'selected_predictions.npz',directory/'summary.json',r,exp['selected_sha256'],train_sha)
    else:add(cfg,directory/'selected.pt',directory/'selected_predictions.npz',directory/'summary.json',ref,row.get('accepted_hashes',{}).get('selected.pt'))
   except Exception as e:issues.append(dict(reference_id=ref['reference_id'],error=repr(e),traceback=traceback.format_exc()))
   if (len(refs)+len(issues))%20==0:write(out/'status.json',dict(state='inventory_in_progress',references=len(refs),unique_model_views=len(models),issues=len(issues),test_inference=False,updated_at=time.time()))
 # Parent pairs are separate references, and are not new training tasks.
 for seed,pair in sorted(parents.items()):
  assert set(pair)=={'cfp','oct'}
  key=digest_json(dict(factory='PilotGraph_independent_parent_pair',seed=seed,parents=canonical(pair),evaluation_state='eval',preprocessing='CFP224_fullspan32_OCT96_v1'))
  for value in pair.values():assert sha(value['path'])==value['sha256']
  models[key]=dict(model_view_id=key,factory='PilotGraph_independent_parent_pair',seed=seed,parent_checkpoints=pair,
   reference_ids=[f'parent_seed{seed}'],strict_model_load_verified=False,development_forward_replay_verified=False,test_inference_permitted=False)
  refs.append(dict(reference_id=f'parent_seed{seed}',source_group='independent_parents',model_view_id=key,regime='independent_parent_pair'))
 write(out/'candidate_models.json',list(models.values()));write(out/'reference_mapping.json',refs);write(out/'issues.json',issues);write(out/'sources.json',sources)
 write(out/'status.json',dict(state='candidate_inventory_needs_review' if issues else 'candidate_artifacts_verified_test_still_sealed',
    training_references=sum(x['training_references'] for x in sources),expanded_references=len(refs),unique_model_views=len(models),parent_pair_views=len(parents),
    issues=len(issues),models_sha256=sha(out/'candidate_models.json'),reference_mapping_sha256=sha(out/'reference_mapping.json'),
    development_participants=len(anchor[0]) if anchor else None,strict_model_load=False,development_model_forward_replay=False,statistical_comparison_lock=False,
    historical_test_access_audit=False,final_registry_locked=False,test_inference=False,elapsed_seconds=time.time()-start,updated_at=time.time()))
 print(json.dumps(read(out/'status.json')),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
