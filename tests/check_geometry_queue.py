"""Matrix, reuse identities, fixed-host augmentation, and contrast weights."""
import copy,json
from radonbridge.geometry_study import *
from scripts.report_geometry_mechanism import definitions

parents={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp','oct')} for s in SEEDS}
bases={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in NODES} for s in SEEDS}
rows=direct_rows(parents,bases);cat=mechanism_catalog()
assert len(rows)==684 and len(cat['structures'])==48
assert cat['execution_authorized'] and not catalog()['execution_authorized']
for r in rows:
    r['state']='accepted';c=copy.deepcopy(r['configuration'])
    for ref in c['parent_checkpoints'].values():ref['path']='/relocated'
    assert fingerprint(c)==r['fingerprint']
    if r['structure'].get('M'):
        b=c['bridges'][0];assert b['r']*b['M']==b['h'] and b['r']/256==b['rho']
        c['bridges'][0]['S']+=1;assert fingerprint(c)!=r['fingerprint']
for host in list(rows):
    if host['structure']['id'] not in FIXED_HOSTS:continue
    for arm in ('continue','radon','linear_resample'):
        c=augmentation_configuration(host,{'path':'/host','sha256':'abc'},bases[host['seed']],arm)
        assert c['bridges'][0]==host['configuration']['bridges'][0]
        assert len(c['bridges'])==(1 if arm=='continue' else 2)
        if arm!='continue':assert c['bridges'][1]['h']==512 and c['bridges'][1]['parallel_to']==0
        rows.append(dict(host,category='augmentation',structure=dict(id=host['structure']['id']+'_plus_'+arm)))
assert len(rows)==756
defs=definitions(rows,cat);assert len(defs)==64
for d in defs:
    if d['estimable']:
        w=d['weights'];assert abs(w.sum())<1e-12 and abs(w[w>0].sum()-1)<1e-12 and abs(w[w<0].sum()+1)<1e-12
assert sum(not d['estimable'] for d in defs)==2  # Low-budget k5 arithmetic infeasible, both protocols.
rows[0]['state']='infeasible'
assert sum(not d['estimable'] for d in definitions(rows,cat))>2
print(json.dumps(dict(passed=True,direct=684,total=756,structures=48,contrasts=64,checks=['dedup','path_identity','fixed_host','paired_weights','infeasible_not_performance'])))

# Regression: completed/failed segment controllers must not run exit callbacks
# after full archiving has retired their working directories.
import tempfile,signal
from pathlib import Path
from unittest.mock import patch,Mock
from scripts.run_geometry_mechanism import Queue
for fail in (False,True):
    with tempfile.TemporaryDirectory() as tmp:
        q=Queue.__new__(Queue);q.root=Path(tmp);q.p={};q.commit='test';q.status=Mock()
        c=Mock();c.run_jobs.side_effect=RuntimeError('injected') if fail else None
        c.run_jobs.return_value={'job':{'passed':True}}
        oldterm=signal.getsignal(signal.SIGTERM);oldint=signal.getsignal(signal.SIGINT)
        with patch('scripts.run_geometry_mechanism.Controller',return_value=c),patch('scripts.run_geometry_mechanism.storage_check'),patch('scripts.run_geometry_mechanism.atexit.unregister') as unregister:
            try:q.execute([], 'preflight','fixture')
            except RuntimeError:assert fail
            else:assert not fail
            c.shutdown.assert_called_once();unregister.assert_called_once_with(c.shutdown)
        signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGINT,oldint)
print(json.dumps(dict(passed=True,exit_callback_regression=True)))
# A serialized immutable protocol must be identical after restart (tuple/list normalization).
with tempfile.TemporaryDirectory() as tmp:
    first=Queue(Path(tmp),'same-commit')
    second=Queue(Path(tmp),'same-commit')
    assert first.p==second.p
print(json.dumps(dict(passed=True,protocol_restart_roundtrip=True)))
