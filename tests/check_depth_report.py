"""Synthetic participant bootstrap, coverage, zero/missing estimands and safe output."""
import json,tempfile
from pathlib import Path
import numpy as np
from radonbridge.depth_study import matrices,definitions,SEEDS,LRS
from radonbridge.geometry_study import configuration
from radonbridge.pretest_study import row
from radonbridge.artifacts import sha256
from scripts.geometry_evidence import write,read
from scripts.report_depth_study import build

with tempfile.TemporaryDirectory() as t:
    root=Path(t)
    parents={s:{k:dict(path='/synthetic',sha256=str(s)+k) for k in ('cfp','oct')} for s in SEEDS}
    bases={s:{f'{b}_stage{stage}':dict(path='/synthetic',sha256=str(s)+b+str(stage)) for b in ('cfp','oct') for stage in (2,3,4)} for s in SEEDS}
    rows=matrices(parents,bases)
    baseline=[row(f'none_{s}_{lr}',configuration(dict(family='none'),s,lr,'branch',parents[s],{}),dict(id='no_communication'),'baseline') for s in SEEDS for lr in LRS]
    definitions_=definitions(rows,baseline);rng=np.random.default_rng(78);y=np.tile(np.arange(2),148);ids=np.asarray([f'synthetic{i}' for i in range(296)])
    for r in rows+baseline:
        d=root/r['id'];d.mkdir();v={}
        for k in ('cfp','oct'):
            pred=y.copy();flip=rng.choice(296,size=80,replace=False);pred[flip]=1-pred[flip]
            v[k]=np.eye(2)[pred]*.8+.1
        np.savez(d/'selected_predictions.npz',ids=ids,y=y,**v)
        r.update(directory=str(d),state='accepted',accepted_hashes={'selected_predictions.npz':sha256(d/'selected_predictions.npz')})
    reused=[r for r in rows if r['structure']['stages']==[3] and r['structure']['r']==16]
    pending=[r for r in rows if r not in reused];assert len(pending)==96
    pending[0]['state']='infeasible'
    write(root/'manifest.json',dict(rows=pending));write(root/'references.json',dict(single_stage3=reused,no_communication=baseline));write(root/'comparisons.json',definitions_)
    build(root)
    stats=read(root/'report/statistics.json');assert stats['primary_comparisons']==22 and stats['participants']==296 and stats['resamples']==10000
    assert len(stats['contrasts'])==60 and any(not d['estimable'] for d in stats['contrasts'])
    assert any(d.get('family_simultaneous_ci95_pp') is not None for d in stats['contrasts'])
    assert all('synthetic' not in p.read_text() for p in (root/'report').iterdir())
    assert read(root/'comparisons.json')==definitions_,'Reporting must not reweight missing cells'
print(json.dumps(dict(passed=True,synthetic=True,shared_bootstrap=10000,missing_cells_not_reweighted=True,primary=22,secondary=38,participant_ids_not_exported=True)))
