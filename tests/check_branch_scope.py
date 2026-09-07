"""Scope withdrawal preserves branch configs and never creates a terminal head."""
import copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
from radonbridge.geometry_study import direct_rows,SEEDS,FIXED_HOSTS,augmentation_configuration,fingerprint
from scripts.run_branch_mechanism import branch_catalog,assert_branch_rows,BranchQueue
from scripts.report_geometry_mechanism import definitions
from scripts.geometry_evidence import write
from radonbridge.artifacts import sha256

parents={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp','oct')} for s in SEEDS}
bases={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp_stage3','oct_stage3')} for s in SEEDS}
all_rows=direct_rows(parents,bases)
rows=[copy.deepcopy(r) for r in all_rows if r['protocol']=='branch']
assert len(rows)==342
hosts=[r for r in rows if r['structure']['id'] in FIXED_HOSTS]
assert len(hosts)==12
for host in hosts:
    for arm in ('continue','radon','linear_resample'):
        cfg=augmentation_configuration(host,dict(path='/host',sha256='abc'),bases[host['seed']],arm)
        rows.append(dict(host,configuration=cfg,fingerprint=fingerprint(cfg),category='augmentation',
                         structure=dict(id=host['structure']['id']+'_plus_'+arm)))
assert len(rows)==378
assert_branch_rows(rows)
for row in rows:row['state']='accepted'
cat=branch_catalog();defs=definitions(rows,cat)
assert len(defs)==32 and sum(d['family']=='direct28' for d in defs)==28
assert sum(d['family']=='augmentation4' for d in defs)==4
assert sum(not d['estimable'] for d in defs)==1
for d in defs:
    if d['estimable']:
        w=d['weights'];assert abs(w.sum())<1e-12 and abs(w[w>0].sum()-1)<1e-12
for bad in [next(r for r in all_rows if r['protocol']=='fusion'),dict(rows[0],configuration=dict(rows[0]['configuration'],task_fusion={'pooling':'mean'}))]:
    try:assert_branch_rows([bad])
    except AssertionError:pass
    else:raise AssertionError('Fusion must be rejected')
assert json.loads(json.dumps(cat))==cat
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);snap=root/'source_snapshot'
    write(snap/'protocol.json',dict(source_commit='old'))
    write(snap/'snapshot_hashes.json',{'protocol.json':sha256(snap/'protocol.json')})
    first=BranchQueue(root,'new');second=BranchQueue(root,'new');assert first.p==second.p
    write(snap/'protocol.json',{'tampered':True})
    try:BranchQueue(root,'new')
    except AssertionError:pass
    else:raise AssertionError('Snapshot mutation must be rejected')
print(json.dumps(dict(passed=True,direct=342,augmentation=36,total=378,comparisons=32,
 checks=['branch_fingerprints_preserved','no_terminal_fusion','fixed_hosts','comparison_weights','snapshot_integrity','restart_roundtrip'])))
