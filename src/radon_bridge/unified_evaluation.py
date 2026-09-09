"""Locked read-only evaluation. Training entry points retain their test prohibition."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from .artifacts import resolve, relocate, sha256
from .component_ablation import component_switch, NAMES, STATES, predict
from .communication_analysis import paired, derangements
from .data import PairedDataset
from .development_replay import state_hash, compare_arrays
from .diagnostics import read_only, release_forward_graph
from .experiment import evaluate, write_json
from .metrics import classification_metrics
from .model import PilotGraph


def verify_lock(lock, split):
    candidate=json.loads((lock/'candidate_lock.json').read_text())
    for name,digest in candidate['files'].items():
        if sha256(lock/name)!=digest:raise ValueError('Locked input changed: '+name)
    if split=='test':
        seal=json.loads((lock/'test_authorization.json').read_text())
        if not seal['test_inference_permitted'] or seal['candidate_lock_sha256']!=sha256(lock/'candidate_lock.json'):
            raise ValueError('Test authorization invalid')
        if seal['source_commit']!=os.environ.get('RB_EVALUATION_COMMIT'):
            raise ValueError('Test executable differs from accepted source')
        for path,digest in seal['preflight_summaries'].items():
            if sha256(path)!=digest:raise ValueError('Preflight evidence changed')
    elif split!='validation':raise ValueError('Only preflight development or locked test is supported')
    return candidate


def load_model(record):
    if record.get('factory')=='PilotGraph_independent_parent_pair':
        g=PilotGraph(seed=record['seed'],bridge_configs=[],device='cuda')
        for branch,ref in record['parent_checkpoints'].items():
            if sha256(ref['path'])!=ref['sha256']:raise ValueError('Parent SHA changed')
            s=torch.load(resolve(ref['path']),map_location='cpu',weights_only=False)
            if s['branch']!=branch or s['seed']!=record['seed'] or s['training_stage']!='independent':raise ValueError('Parent identity mismatch')
            g.load_native_state(s['model'],branch=branch)
    else:
        c=relocate(record['configuration'])
        if c.get('task_fusion'):raise ValueError('Withdrawn learned fusion is excluded')
        if sha256(record['checkpoint_path'])!=record['checkpoint_sha256']:raise ValueError('Model SHA changed')
        g=PilotGraph(seed=c['seed'],bridge_configs=c['bridges'],device='cuda')
        s=torch.load(resolve(record['checkpoint_path']),map_location='cpu',weights_only=False)
        g.load_complete_state(s['model'])
    g.graph.eval()
    return g


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in ('ids','y','cfp','oct')}


def validate_prediction(a, data):
    expected=np.asarray([r['order'] for r in data.rows]); labels=np.asarray([int(r['label_id']) for r in data.rows])
    if not np.array_equal(a['ids'],expected) or not np.array_equal(a['y'],labels):raise ValueError('Participant or label order changed')
    if len(set(expected))!=len(data):raise ValueError('Duplicate participant')
    for branch in ('cfp','oct'):
        p=a[branch]
        if p.shape!=(len(data),2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1.,atol=1e-5):
            raise ValueError('Invalid probabilities')


def run(job,lock,out,split):
    cfg=verify_lock(lock,split);start=time.monotonic()
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    root=Path(cfg['data_directory'] if split=='test' else cfg['development_data_directory'])
    expected_sha=json.loads((lock/'data_acceptance.json').read_text())['selected_sha256'] if split=='test' else '4285250208dd1496a6912ed6e072abf9d2aa87e05216e78be6ad405e9ea7d24c'
    if sha256(root/'selected.csv')!=expected_sha:raise ValueError('Evaluation cohort changed')
    data=PairedDataset(str(root),split,224)
    if len(data)!=(290 if split=='test' else 296):raise ValueError('Unexpected participant count')
    g=load_model(job['model']);before=state_hash(g);node_ids={n.name:n.id for n in g.nodes}
    views={v['view_id']:v for v in json.loads((lock/'model_views.json').read_text())}
    mid=job['model']['model_view_id'];files=[];detail={}
    with read_only(g):
        evaluate(g,data,16,job['model'].get('seed',job['model'].get('configuration',{}).get('seed')),out/'predictions.npz')
        baseline=arrays(out/'predictions.npz');validate_prediction(baseline,data)
        if split=='validation':compare_arrays(baseline,arrays(views[mid]['development_prediction']))
        files.append('predictions.npz')
        if job['kind']=='component':
            for state,(host_on,new_on) in zip(NAMES,STATES):
                with component_switch(g,host_on,new_on):value=predict(g,data)
                validate_prediction(value,data);np.savez(out/(state+'.npz'),**value);files.append(state+'.npz')
                if split=='validation':
                    view=mid if state=='both_on' else mid+'__'+state
                    compare_arrays(value,arrays(views[view]['development_prediction']))
                if state=='both_on':
                    for b in ('cfp','oct'):
                        if not np.allclose(value[b],baseline[b],rtol=1e-5,atol=1e-6):raise ValueError('Both-on differs from complete forward')
            replay=predict(g,data)
            if any(not np.array_equal(replay[k],baseline[k]) for k in baseline):raise ValueError('Component hook restoration failed')
            native_state={k:v for k,v in g.save_state().items() if k in g.definition.native_checkpoint_modules}
            # Same updated native modules, independent no-bridge topology.
            native=PilotGraph(seed=job['model']['configuration']['seed'],bridge_configs=[],device='cpu')
            native.load_native_state(native_state)
            native.graph.to('cuda')
            with read_only(native):native_pred=predict(native,data)
            off=arrays(out/'both_off.npz')
            if any(not np.allclose(off[b],native_pred[b],atol=1e-6,rtol=1e-5) for b in ('cfp','oct')):raise ValueError('Both-off differs from updated native network')
            del native
            detail['both_off_matches_updated_native']=True
        elif job['kind']=='pairing':
            permutation_file=lock/'test_permutations.npz'
            if split=='validation':
                permutation_file=out/'development_permutations.npz'
                np.savez(permutation_file,ids=baseline['ids'],permutations=derangements(len(data)))
            detail['pairing']=paired(g,data,out,permutation_file,out/'predictions.npz')
            files.append('private_perturbation_predictions.npz')
        elif job['kind']!='model':raise ValueError('Unknown locked job kind')
        release_forward_graph(g)
    if state_hash(g)!=before or node_ids!={n.name:n.id for n in g.nodes}:raise ValueError('Model state or node identity changed')
    metrics={b:classification_metrics(baseline['y'],baseline[b]) for b in ('cfp','oct')}
    write_json(out/'summary.json',dict(state='accepted',job_id=job['job_id'],kind=job['kind'],model_view_id=mid,
        split=split,participants=len(data),batch=16,source_commit=os.environ.get('RB_EVALUATION_COMMIT'),
        strict_model_load=True,parameters_BN_gradients_RNG_preserved=True,original_node_ids_preserved=True,
        development_match=split=='validation',test_used=split=='test',no_training=True,metrics=metrics,detail=detail,
        prediction_files={name:sha256(out/name) for name in files},record_sha256=sha256(out/'record.json'),
        candidate_lock_sha256=sha256(lock/'candidate_lock.json'),peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        seconds=time.monotonic()-start,updated_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('record','lock','output'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--split',choices=('validation','test'),required=True);a=p.parse_args()
    try:
        if (a.output/'summary.json').exists():raise ValueError('Refuse to overwrite accepted inference')
        job=json.loads(a.record.read_text())
        all_jobs=json.loads((a.lock/'jobs.json').read_text())
        if job not in all_jobs:raise ValueError('Job not in locked registry')
        run(job,a.lock,a.output,a.split)
    except BaseException:
        write_json(a.output/'failure.json',dict(state='needs_attention',error=traceback.format_exc(),split=a.split,test_used=a.split=='test'))
        raise
