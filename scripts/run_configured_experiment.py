"""Run explicit matched comparison groups with accepted source and one budget."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(p):
    from scripts.run_integer_experiment import source_hashes
    assert source_hashes() == p['accepted_source_hashes'], 'Accepted source changed'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == p['source_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    if p.get('gpu_time_policy') == 'unlimited_until_convergence':
        assert p['max_gpu_minutes'] is None
    else:
        assert p['prior_gpu_minutes'] + p['max_gpu_minutes'] <= 240
    assert sha(__file__) == p['driver_sha256']
    for item in p['references'].values():
        assert sha(Path(item['directory']) / 'summary.json') == item['summary_sha256']
    for group in p['groups']:
        for j in group['jobs']:
            for parent in j['config']['parent_checkpoints'].values():
                assert sha(parent['path']) == parent['sha256']
            for bridge in j['config']['bridges']:
                if bridge.get('compression') == 'fixed_svd_channel':
                    from radonbridge.svd_basis import _load_basis
                    for key, artifact in bridge['basis_files'].items():
                        assert sha(artifact['path']) == artifact['sha256']
                        _, _, metadata = _load_basis(artifact['path'], artifact['sha256'])
                        assert metadata['source_key'] == key and metadata['seed'] == j['config']['seed']
                        assert metadata['provenance']['parent_checkpoints'] == j['config']['parent_checkpoints']


def bootstrap(root, protocol):
    import numpy as np
    from radonbridge.experiment import write_json
    def f1(y, pred):
        vals=[]
        for c in (0, 1):
            tp=((y == c) & (pred == c)).sum(axis=-1)
            den=(y == c).sum(axis=-1)+(pred == c).sum(axis=-1)
            vals.append(np.divide(2*tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0))
        return sum(vals)/2
    def pair(a, b):
        with np.load(a/'selected_predictions.npz', allow_pickle=False) as aa, np.load(b/'selected_predictions.npz', allow_pickle=False) as bb:
            assert np.array_equal(aa['ids'], bb['ids']) and np.array_equal(aa['y'], bb['y'])
            assert len(np.unique(aa['ids'])) == len(aa['ids']), 'Bootstrap requires unique participants'
            y=aa['y']; ap={k:aa[k].argmax(1) for k in ('cfp','oct')}; bp={k:bb[k].argmax(1) for k in ap}
        rng=np.random.default_rng(20260904); deltas={k:[] for k in ap}
        for _ in range(100):
            idx=rng.integers(0, len(y), size=(100, len(y)))
            for k in ap:
                deltas[k].extend((f1(y[idx], bp[k][idx])-f1(y[idx], ap[k][idx])).tolist())
        delta={k:float(f1(y,bp[k])-f1(y,ap[k])) for k in ap}
        deltas['mean']=(np.asarray(deltas['cfp'])+np.asarray(deltas['oct']))/2
        delta['mean']=(delta['cfp']+delta['oct'])/2
        return {k:{'delta_pp':100*delta[k], 'ci95_pp':(100*np.quantile(v,[.025,.975])).tolist()} for k,v in deltas.items()}
    comparisons={name:(root/pair[0],root/pair[1]) for name,pair in protocol['bootstrap_pairs'].items()}
    result={'resamples':10000,'unit':'paired participant','test_used':False,
            'interpretation':'Exploratory development-set intervals conditional on selected checkpoints; do not correct selection bias.', 'comparisons':{}}
    for name,(a,b) in comparisons.items():
        if all((x/'summary.json').exists() and read(x/'summary.json').get('converged_by_policy') for x in (a,b)):
            result['comparisons'][name]=pair(a,b)
    write_json(root/'bootstrap.json', result)


def run(root, p):
    from scripts.run_integer_experiment import Controller
    from radonbridge.experiment import write_json
    verify(p)
    for name,item in p['references'].items():
        (root/name).symlink_to(item['directory'], target_is_directory=True)
    args=SimpleNamespace(output=str(root),protocol=str(root/'protocol.json'),phase='run',data=p['data'])
    c=Controller(args)
    report={'trials':{},'test_used':False,'source_commit':p['source_commit'], 'groups':[]}
    try:
        bootstrap(root,p)
        for group in p['groups']:
            verify(p)
            c.phase=group['name']
            if not c.group_fits(group['jobs'], group['estimate_seconds_per_epoch']):
                report['groups'].append({'name':group['name'],'state':'budget_blocked'})
                break
            completed=c.run_jobs(group['jobs'])
            report['trials'].update(completed)
            assert len({v['initial_native_sha256'] for v in completed.values()}) == 1
            good=all(v['converged_by_policy'] for v in completed.values())
            report['groups'].append({'name':group['name'],'state':'complete' if good else 'incomplete'})
            write_json(root/'study_summary.json',report)
            bootstrap(root,p)
            if not good:
                c.status('needs_attention',reason='Trial reached epoch cap without validation plateau')
                break
        else:
            c.status('complete')
    except InterruptedError:
        report['interrupted']=True
    except Exception:
        c.status('failed',reason=traceback.format_exc())
        raise
    finally:
        c.shutdown()
        report['new_gpu_minutes']=c.used()
        report['cumulative_gpu_minutes']=p['prior_gpu_minutes']+c.used()
        write_json(root/'study_summary.json',report)
        subprocess.run([os.sys.executable,'scripts/summarize_integer_experiment.py','--root',str(root)],check=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();root=Path(args.root);p=read(root/'protocol.json')
    if args.check:
        verify(p)
        print(json.dumps({'verified':True,'groups':[g['name'] for g in p['groups']], 'budget':p['max_gpu_minutes']}))
    else:
        with (root.parent.parent/'.active.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                run(root,p)
            except Exception:
                from radonbridge.experiment import write_json
                write_json(root/'controller_failure.json',{'error':traceback.format_exc(),'time':time.time()})
                raise
