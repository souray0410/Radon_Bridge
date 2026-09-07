"""Bounded metadata-only audit of prior LOOK participant roles; never infer test performance."""
import argparse,csv,hashlib,json,time
from pathlib import Path
from scripts.geometry_evidence import write

TARGET=Path('/data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/glaucoma_all_evidence/primary/reference_labels.csv')
EXPECTED='eae604091a1094ed70ff4edbcf6e59d00b57b5be124fb4ac5f76ece1beac0d8d'
HISTORICAL=Path('/data/mengh/LOOK/2026_09_03_08_30_00/runs/backbones')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(out):
 assert not out.exists();out.mkdir(parents=True)
 assert sha(TARGET)==EXPECTED
 target={r['participant_id'] for r in csv.DictReader(TARGET.open()) if r['split']=='test'};assert len(target)==290
 evidence=[];role_sets={'train':set(),'validation':set(),'test':set()};cache={};skipped=[]
 for p in sorted(HISTORICAL.glob('*/run_config.json')):
  d=json.loads(p.read_text());mark=p.parent/'training_complete.json'
  if not mark.exists():skipped.append(dict(config=str(p),reason='no completion marker'));continue
  complete=json.loads(mark.read_text())
  if complete.get('status')!='complete' or complete.get('epochs_completed',0)<=0:continue
  labels=Path(d['config']['labels_csv'])
  if labels not in cache:cache[labels]=(sha(labels),list(csv.DictReader(labels.open())))
  digest,rows=cache[labels];assert digest==d['labels_sha256']==complete['labels_sha256']
  checkpoint=Path(complete['portable_checkpoint']);assert checkpoint.is_file()
  roles={s:{r['participant_id'] for r in rows if r['split']==s}&target for s in role_sets}
  for k,v in roles.items():role_sets[k].update(v)
  evidence.append(dict(run_config=str(p),run_config_sha256=sha(p),completion_marker=str(mark),completion_sha256=sha(mark),
   historical_labels_path=str(labels),historical_labels_sha256=digest,epochs_completed=complete['epochs_completed'],
   selection_criterion=complete.get('criteria'),checkpoint_exists=True,checkpoint_sha256_recorded=complete['checkpoint_sha256'],
   current_test_participants_by_historical_role={k:len(v) for k,v in roles.items()}))
 overlap=role_sets['train']|role_sets['validation']
 result=dict(state='hold_test_unsealing_pending_lineage_review' if overlap else 'no_overlap_in_this_bounded_scope_not_global_clearance',
  target_test_participants=290,target_labels_sha256=EXPECTED,completed_historical_runs_checked=len(evidence),
  unique_overlap_by_historical_role={k:len(v) for k,v in role_sets.items()},unique_prior_train_or_development_participants=len(overlap),
  scope=str(HISTORICAL),audit_exhaustive=False,checkpoint_weight_inheritance_checked=False,research_selection_influence_resolved=False,
  test_inference=False,test_performance_read=False,participant_identifiers_exported=False,
  interpretation='Prior related-study exposure established by matched label SHA and completed training records. Direct R&B weight leakage is not established. Hold independent-test claims and unsealing until lineage and research-use review; do not automatically exclude participants or change the split.',updated_at=time.time())
 write(out/'evidence.json',evidence);write(out/'audit.json',result);write(out/'skipped.json',skipped)
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
