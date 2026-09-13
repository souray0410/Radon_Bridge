"""Accepted-parent -> fixed bases -> finite matched trials -> diagnostics/report."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash
from radon_bridge.models.native_materialization import verify_selected,load_selected
from radon_bridge.data.observed_pair import from_parent_specs,collate_observed
from radon_bridge.evaluation.native_replay import replay_selected
from radon_bridge.evaluation.paired_native import evaluate,replay_matches
from radon_bridge.methods.native_basis import fit
from radon_bridge.studies.project_build import build
from radon_bridge.studies.research_matrix import arms
from radon_bridge.training.paired_native import train,validate


def read(p):return json.loads(Path(p).read_text())


def verify_arm(root,identity):
    root=Path(root);r=read(root/'accepted.json')
    if r.get('state')!='accepted' or r.get('identity')!=identity or not r.get('plateau') or r.get('test_access') is not False:raise ValueError('Arm receipt mismatch')
    required={'best.pt','last.pt','development_predictions.npz','history.json','replay_predictions.npz'}
    if not required.issubset(r['files']):raise ValueError('Incomplete trial evidence')
    for n,digest in r['files'].items():
        p=(root/n).resolve()
        if not p.is_relative_to(root.resolve()) or file_sha256(p)!=digest:raise ValueError('Arm artifact changed')
    return r


def validate_spec(spec):
    if spec.get('schema')=='radon_group_case_v1':
        from radon_bridge.studies.group_case import validate_spec as validate_group
        return validate_group(spec)
    if spec.get('schema')!='radon_project_case_v1' or spec.get('test_access') is not False:raise ValueError('Unknown/sealed project case')
    if spec['arms']!=arms(spec['disease'],spec['model']['name']) or spec['seed'] not in (3416,3417,3418):raise ValueError('Finite scientific matrix changed')
    validate(spec['training'])
    for role,track in [('cfp','cfp_2d'),('oct','oct_volume_3d')]:
        ref=spec['parents'][role]
        if file_sha256(Path(ref['path'])/'selected_artifact.json')!=ref['manifest_sha256']:raise ValueError('Parent receipt changed')
        _,p,_=verify_selected(ref['path'])
        if p['training']['seed']!=spec['seed'] or p['track']!=track or p['disease']!=spec['disease']:raise ValueError('Parent identity mismatch')
    for ref in spec['source_pins']:
        if file_sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Immutable execution source changed')


def verify_case(root,spec):
    root=Path(root);r=read(root/'accepted.json')
    if r.get('schema')!=spec['schema'] or r.get('state')!='accepted' or r.get('identity')!=stable_hash(spec) or r.get('test_access') is not False:raise ValueError('Case identity not accepted')
    if len(r['arms'])!=len(spec['arms']):raise ValueError('Missing matched arms')
    for name,identity in r['arms'].items():verify_arm(root/'arms'/name,identity)
    for n,digest in r['files'].items():
        p=(root/n).resolve()
        if not p.is_relative_to(root.resolve()) or file_sha256(p)!=digest:raise ValueError('Case evidence changed')
    if not {'spec.json','bases/accepted.json','report/paired_statistics.json','report/metrics.csv','diagnostics/accepted.json'}.issubset(r['files']):raise ValueError('Incomplete report/diagnostic evidence')
    diagnostic=read(root/'diagnostics/accepted.json')
    if diagnostic.get('identity')!=stable_hash(spec) or diagnostic.get('state')!='accepted':raise ValueError('Diagnostic identity changed')
    for n,digest in diagnostic['files'].items():
        p=(root/'diagnostics'/n).resolve()
        if not p.is_relative_to((root/'diagnostics').resolve()) or file_sha256(p)!=digest:raise ValueError('Diagnostic artifact changed')
    return r


def prepare(spec,out,device,should_pause):
    if spec.get('schema')=='radon_group_case_v1':
        from radon_bridge.studies.group_case import prepare as prepare_group
        return prepare_group(spec,out,device,should_pause)
    from mhd_framework.models import create_model
    from expanded.native import Inputs,collate
    validate_spec(spec);parents={};ps={}
    for role in ('cfp','oct'):
        root=Path(spec['parents'][role]['path']);ps[role]=read(root/'spec.json')
        model=load_selected(root,create_model,device='cpu',allow_inference_equivalence=True)
        replay=out/'parents'/role/'replay.json'
        if replay.exists():
            r=read(replay)
            if r['best_sha256']!=file_sha256(root/'best.pt') or r.get('inference_compatibility')!=model.inference_compatibility or r.get('status')!='accepted':raise ValueError('Replay evidence changed')
        else:
            if should_pause():raise InterruptedError('Pause before parent replay')
            model.to(device)
            try:replay_selected(model,root,Inputs,collate,device,1,replay)
            finally:model.cpu()
        parents[role]=model
    train=from_parent_specs(ps['cfp'],ps['oct'],'train',Inputs,augment=True,seed=spec['seed'])
    fitting=from_parent_specs(ps['cfp'],ps['oct'],'train',Inputs,seed=spec['seed'])
    dev=from_parent_specs(ps['cfp'],ps['oct'],'development',Inputs,seed=spec['seed'])
    if set(train.participant_ids)&set(dev.participant_ids):raise ValueError('Participant split overlap')
    sample=fitting[0];shapes={k:tuple(sample[k].shape[1:]) for k in ('cfp','oct')}
    return parents,shapes,train,fitting,dev


def execute(spec,output,device='cuda:0'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);stopped=False;device=torch.device(device)
    builder = build
    if spec.get('schema')=='radon_group_case_v1':
        from radon_bridge.studies.group_build import build as group_builder
        def builder(parents, shapes, arm, bases, seed, device):
            model=group_builder(parents,shapes,spec['sources'],arm,bases,seed,device='cpu')
            if spec.get('device_placement'):
                from radon_bridge.models.placement import place
                return place(model, **spec['device_placement'])
            return model.to(device)

    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    def paused():return stopped or (out/'pause.json').exists()
    def status(stage):atomic_write_json(dict(state='running',stage=stage,pid=os.getpid(),updated_at=time.time(),test_access=False),out/'status.json')
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'accepted.json').exists():return verify_case(out,spec)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
        if (out/'spec.json').exists() and read(out/'spec.json')!=spec:raise ValueError('Run specification changed')
        atomic_write_json(spec,out/'spec.json');identity=stable_hash(spec)
        try:
            status('parent_replay');parents,shapes,train_data,fit_data,dev=prepare(spec,out,device,paused)
            status('train_only_basis')
            baseline=builder(parents,shapes,spec['arms'][0],{},spec['seed'],device)
            stages=sorted({s for a in spec['arms'] for s in a['stages']})
            bases=fit(baseline,fit_data,stages,out/'bases',identity,spec['seed'],device,paused)
            collate_case=getattr(train_data,'collate_fn',collate_observed)
            loader=lambda ds:DataLoader(ds,batch_size=spec['training']['microbatch'],collate_fn=collate_case,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(spec['seed']))
            if not (out/'parent_predictions.npz').exists():evaluate(baseline,loader(dev),device,out/'parent_predictions.npz',paused)
            del baseline;gc.collect()
            accepted={}
            for arm in spec['arms']:
                if paused():raise InterruptedError('Pause before next matched arm')
                name=arm['id'];arm_root=out/'arms'/name
                active_bases=bases
                if arm.get('host'):
                    host_arm=next(a for a in spec['arms'] if a['id']==arm['host'])
                    host_path=out/'arms'/arm['host']/'best.pt'
                    host_identity=stable_hash(dict(case=identity,host=arm['host'],best_sha256=file_sha256(host_path)))
                    if not (out/'host_bases'/arm['host']/'accepted.json').exists():
                        host_model=builder(parents,shapes,host_arm,bases,spec['seed'],device)
                        host_model.load_state_dict(torch.load(host_path,map_location='cpu',weights_only=False)['model'],strict=True)
                        active_bases=fit(host_model,fit_data,[3],out/'host_bases'/arm['host'],host_identity,spec['seed'],device,paused)
                        del host_model;gc.collect()
                    else:
                        record=read(out/'host_bases'/arm['host']/'accepted.json')
                        if record['identity']!=host_identity:raise ValueError('Host basis identity changed')
                        active_bases=record['bases']
                arm_identity=stable_hash(dict(case=identity,arm=arm,bases=active_bases,host_best_sha256=file_sha256(out/'arms'/arm['host']/'best.pt') if arm.get('host') else None))
                accepted[name]=arm_identity
                if (arm_root/'accepted.json').exists():verify_arm(arm_root,arm_identity);continue
                status('training_'+name)
                model=builder(parents,shapes,arm,active_bases,spec['seed'],device)
                if arm.get('host'):
                    host=torch.load(out/'arms'/arm['host']/'best.pt',map_location='cpu',weights_only=False)
                    state=model.state_dict();missing=set(state)-set(host['model'])
                    if any('bridge_1_' not in k and 'bridge_parallel' not in k for k in missing):raise ValueError('Parallel host state changed')
                    extra=set(host['model'])-set(state)
                    if extra:raise ValueError('Intact nonlinear host was removed')
                    state.update(host['model']);model.load_state_dict(state,strict=True)
                if not (arm_root/'last.pt').exists():
                    path=arm_root/'initial_predictions.npz';evaluate(model,loader(dev),device,path,paused)
                    reference=out/'arms'/arm['host']/'development_predictions.npz' if arm.get('host') else out/'parent_predictions.npz'
                    replay_matches(reference,path)
                from radon_bridge.studies.project_profile import profile_model
                full_eyes=[i for i,n in enumerate(train_data.counts) if n==2][:spec['training']['microbatch']]
                if len(full_eyes)!=spec['training']['microbatch']:raise ValueError('No maximum-eye resource fixture')
                resource_batch=collate_case([train_data[i] for i in full_eyes])
                # A resource fixture updates a copy of the current model state,
                # then exactly restores weights, BN and RNG before formal work.
                profile_model(model,resource_batch,spec['training'],arm_identity,
                    arm_root/'resource'/str(time.time_ns()),device)
                result=train(model,train_data,dev,spec['training'],spec['seed'],arm_root,arm_identity,device,paused)
                del model;gc.collect()
                if result.get('state')!='accepted':
                    atomic_write_json(dict(state=result['state'],stage=name,updated_at=time.time(),test_access=False),out/'status.json');return result
            status('read_only_diagnostics')
            if spec.get('schema')=='radon_group_case_v1':
                from radon_bridge.analysis.group_diagnostics import diagnose
            else:
                from radon_bridge.analysis.native_diagnostics import diagnose
            diagnose(spec,out,parents,shapes,bases,fit_data,dev,device,paused)
            status('matched_statistics')
            if spec.get('schema')=='radon_group_case_v1':
                from radon_bridge.analysis.group_report import report_case
            else:
                from radon_bridge.analysis.project_report import report_case
            report_case(spec,out)
            paths=['spec.json','bases/accepted.json','parent_predictions.npz','diagnostics/accepted.json','report/paired_statistics.json','report/metrics.csv']
            receipt=dict(schema=spec['schema'],state='accepted',identity=identity,arms=accepted,test_access=False,
                files={p:file_sha256(out/p) for p in paths})
            atomic_write_json(receipt,out/'accepted.json');verify_case(out,spec)
            atomic_write_json(dict(state='completed',updated_at=time.time(),test_access=False),out/'status.json');return receipt
        except InterruptedError:
            atomic_write_json(dict(state='paused',updated_at=time.time(),test_access=False),out/'status.json');return {'state':'paused'}
        except Exception as e:
            atomic_write_json(dict(state='needs_review',error=repr(e),updated_at=time.time(),test_access=False),out/'status.json');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();execute(read(a.spec),a.output,a.device)
if __name__=='__main__':main()
