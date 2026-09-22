"""Explicit migration and loading of an accepted small-cohort augmentation host."""
import argparse
import json
from pathlib import Path
import torch
from radon_bridge.runtime.pilot_checkpoint import read as read_state, save as save_state
from radon_bridge.runtime.state import atomic_write_json
from radon_bridge.studies.cohort_case import sha


def migrate(source, output):
    source,output=Path(source),Path(output)
    a=json.loads((source/'accepted.json').read_text())
    if not a['converged_by_policy'] or a['test_used'] is not False:
        raise ValueError('Unaccepted host')
    for name,digest in a['files'].items():
        if sha(source/name)!=digest:raise ValueError('Host artifact changed')
    cfg=a['configuration']
    if len(cfg['bridges'])!=1 or cfg['bridges'][0].get('family') not in ('mmtm','cross_attention','cmx_frm'):
        raise ValueError('One nonlinear host required')
    state=read_state(source/'best.pt',kind='selected',configuration=cfg)
    output.mkdir(parents=True,exist_ok=False)
    if state['configuration']!=cfg:raise ValueError('Checkpoint config mismatch')
    save_state(dict(schema='radon_cohort_augmentation_host_v2',configuration=cfg,model=state['model']),output/'model.pt',kind='augmentation_host')
    atomic_write_json(dict(schema='radon_cohort_augmentation_host_v2',framework_api='V5',configuration=cfg,
        source_receipt_sha256=sha(source/'accepted.json'),model=dict(path=str(output/'model.pt'),sha256=sha(output/'model.pt')),
        predictions=dict(path=str(source/'selected_predictions.npz'),sha256=a['files']['selected_predictions.npz']),
        selected_epoch=a['best_epoch'],test_used=False),output/'manifest.json')


def load_host(cfg, graph):
    ref=cfg['augmentation_host']
    if sha(ref['path'])!=ref['sha256']:raise ValueError('Host manifest changed')
    m=json.loads(Path(ref['path']).read_text());old=m['configuration']
    if m['schema']!='radon_cohort_augmentation_host_v2' or m.get('framework_api')!='V5' or m['test_used'] is not False:
        raise ValueError('Unknown host contract')
    if cfg['parents']!=old['parents'] or cfg['seed']!=old['seed'] or cfg['bridges'][:1]!=old['bridges']:
        raise ValueError('Unmatched host')
    if len(cfg['bridges']) not in (1,2) or (len(cfg['bridges'])==2 and cfg['bridges'][1].get('parallel_to')!=0):
        raise ValueError('Only a parallel residual addition is registered')
    for key in ('model','predictions'):
        if sha(m[key]['path'])!=m[key]['sha256']:raise ValueError('Host '+key+' changed')
    state=read_state(m['model']['path'],kind='augmentation_host',configuration=old)
    if state['schema']!=m['schema'] or state['configuration']!=old:raise ValueError('Host envelope changed')
    graph.load_complete_state(state['model'],allow_new_bridge=len(cfg['bridges'])==2)
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();migrate(a.source,a.output)
