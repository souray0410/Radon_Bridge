"""Read-only, task-separated diagnostics of initial and selected complete groups."""
import gc
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from radon_bridge.analysis.native_diagnostics import tensor_state, render_sinograms
from radon_bridge.evaluation.paired_native import move, evaluate
from radon_bridge.runtime.host_checkpoint import capture_rng, restore_rng
from radon_bridge.runtime.state import atomic_write_json, file_sha256, stable_hash
from radon_bridge.studies.group_build import build


def disabled_exchange(module, inputs, output):
    return torch.zeros_like(output) if getattr(module,'delta_only',False) else torch.cat([x.flatten(1) for x in inputs],1)


def summarize_model(model, train, output, device, paused):
    if train.split!='train' or train.augment: raise ValueError('Read-only training features required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    before=tensor_state(model); rng=capture_rng();mode=model.training
    gradients_before={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    model.eval();keys=model.source_keys
    modules=model.task.modules_by_name();old_scale=modules['task_loss_sum'].scale
    exchanges={n:m for n,m in modules.items() if n.endswith('_exchange')}
    sums={n:np.zeros((len(keys),4)) for n in exchanges}
    probe=sorted(range(len(train)),key=lambda i:train.participant_ids[i])[:128]
    atomic_write_json(dict(split='train',ids=[train.participant_ids[i] for i in probe]),out/'probe.json')
    try:
        with torch.no_grad():
            loader=DataLoader(train,batch_size=1,shuffle=False,collate_fn=train.collate_fn,num_workers=0,generator=torch.Generator().manual_seed(0))
            for index,batch in enumerate(loader):
                if paused():raise InterruptedError('Pause training feature diagnosis')
                model(move(batch,device))
                for name,outer in exchanges.items():
                    ex=getattr(outer,'exchange',outer)
                    for i,(x,d) in enumerate(zip(ex.latest_inputs,ex.latest_deltas)):
                        sums[name][i,0]+=float(x.double().square().sum())
                        sums[name][i,1]+=float(d.double().square().sum())
                        if hasattr(ex,'channel_bases'):
                            sums[name][i,2]+=float(ex.channel_bases[i].encode(x).double().square().sum())
                            sums[name][i,3]+=float(x.double().square().sum())
                            if index==probe[0] and ex.mode=='radon':
                                p=ex.projectors[i](ex.channel_bases[i].encode(x))
                                image=p.reshape(len(x),-1,ex.projectors[i].M,ex.S).square().mean((0,1)).sqrt().cpu().numpy()
                                np.savez(out/f'{name}_{i}_sinogram.npz',projection_rms=image,
                                    directions=np.asarray(ex.projectors[i].metadata['directions']),support=ex.projectors[i].metadata['support'])
        # Group native modules by source stage3; avoid calling all layer norms a stage3.
        selected={}
        for name,module in modules.items():
            if name.endswith('_exchange') or any(name.startswith(k+'_stage3') or name.startswith(k+'_denseblock3') for k in keys):
                if sum(p.numel() for p in module.parameters()): selected[name]=module
        per_target=[];loss=modules['task_loss_sum'];old_weights=loss.weights
        try:
            for target in range(len(keys)):
                loss.weights=tuple(float(i==target) for i in range(len(keys)));model.zero_grad(set_to_none=True)
                for index in probe:
                    if paused():raise InterruptedError('Pause task gradient diagnosis')
                    batch=train.collate_fn([train[index]])
                    model(move(batch,device),1/len(probe));model.backward()
                per_target.append({name:torch.cat([p.grad.detach().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for p in module.parameters()]) for name,module in selected.items()})
        finally:loss.weights=old_weights
        gradient=[]
        for left,right in itertools.combinations(range(len(keys)),2):
            for name,a in per_target[left].items():
                b=per_target[right][name];an=float(a.norm());bn=float(b.norm())
                gradient.append(dict(left=keys[left],right=keys[right],module=name,left_norm=an,right_norm=bn,
                    cosine=float(torch.dot(a,b)/(an*bn)) if an and bn else None,
                    undefined_reason=None if an and bn else 'zero_gradient_norm'))
        result=dict(train_participants=len(train),probe_participants=len(probe),probe_sha256=file_sha256(out/'probe.json'),
            residual_ratio={n:{k:float((v[1]/v[0])**.5) if v[0] else None for k,v in zip(keys,values)} for n,values in sums.items()},
            retained_energy={n:{k:float(v[2]/v[3]) if v[3] else None for k,v in zip(keys,values)} for n,values in sums.items()},
            gradients=gradient,probe_microbatch=1,participant_weighting='accumulate CE gradients before cosine',test_access=False)
        if tensor_state(model)!=before:raise ValueError('Diagnosis changed parameters or BN')
        atomic_write_json(result,out/'summary.json');render_sinograms(out)
        return result
    finally:
        for name,p in model.named_parameters():p.grad=gradients_before[name]
        modules['task_loss_sum'].scale=old_scale
        model.train(mode);restore_rng(rng)


def diagnose(spec,root,parents,shapes,bases,train,dev,device,paused):
    root=Path(root);out=root/'diagnostics';out.mkdir(exist_ok=True)
    identity=stable_hash(spec)
    if (out/'accepted.json').exists():
        r=json.loads((out/'accepted.json').read_text())
        if r['identity']!=identity:raise ValueError('Diagnostic identity changed')
        for name,digest in r['files'].items():
            if file_sha256(out/name)!=digest:raise ValueError('Diagnostic evidence changed')
        return r
    for arm in spec['arms']:
        active=bases
        if arm.get('host'):active=json.loads((root/'host_bases'/arm['host']/'accepted.json').read_text())['bases']
        for phase in ('initial','selected'):
            target=out/arm['id']/phase
            if (target/'accepted.json').exists():
                accepted=json.loads((target/'accepted.json').read_text())
                if accepted.get('identity')!=identity:raise ValueError('Diagnostic phase identity changed')
                for name,digest in accepted['files'].items():
                    if file_sha256(target/name)!=digest:raise ValueError('Diagnostic phase evidence changed')
                continue
            model=build(parents,shapes,spec['sources'],arm,active,spec['seed'],device='cpu')
            if spec.get('device_placement'):
                from radon_bridge.models.placement import place
                place(model,**spec['device_placement'])
            else:model.to(device)
            if phase=='selected':
                state=torch.load(root/'arms'/arm['id']/'best.pt',map_location='cpu',weights_only=False)['model']
                model.load_state_dict(state,strict=True)
            elif arm.get('host'):
                state=model.state_dict();state.update(torch.load(root/'arms'/arm['host']/'best.pt',map_location='cpu',weights_only=False)['model'])
                model.load_state_dict(state,strict=True)
            try:
                summarize_model(model,train,target,device,paused)
                if phase=='selected' and arm.get('host') and arm['addition']!='continue':
                    modules=model.task.modules_by_name();states={}
                    for name,disabled in [('both_on',()),('host_only',(1,)),('new_only',(0,)),('both_off',(0,1))]:
                        hooks=[modules[f'bridge_{i}_exchange'].register_forward_hook(disabled_exchange) for i in disabled]
                        try:
                            states[name]=evaluate(model,DataLoader(dev,batch_size=1,collate_fn=dev.collate_fn,num_workers=0,
                                generator=torch.Generator().manual_seed(0)),device,target/(name+'.npz'),paused)
                        finally:
                            for hook in hooks:hook.remove()
                    atomic_write_json(dict(states=states,test_access=False,interpretation='fixed-model deletion; not retraining benefit'),target/'host_states.json')
                if (phase=='selected' and arm['family']=='radon' and arm['mode']=='radon' and
                        arm['compression'] in ('fixed_svd_channel','fixed_random_orthogonal_channel') and
                        len(arm['stages'])==1 and not arm.get('host')):
                    from radon_bridge.analysis.group_pairing import diagnose as pairing_diagnosis
                    before=tensor_state(model);rng=capture_rng()
                    try:pairing_diagnosis(model,dev,target/'pairing',device,paused)
                    finally:restore_rng(rng)
                    if tensor_state(model)!=before:raise ValueError('Pairing changed weights or BN')
                files={str(p.relative_to(target)):file_sha256(p) for p in target.rglob('*') if p.is_file() and p.name!='accepted.json'}
                atomic_write_json(dict(identity=identity,phase=phase,state='accepted',files=files,test_access=False),target/'accepted.json')
            finally:del model;gc.collect()
    files={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='accepted.json'}
    result=dict(identity=identity,state='accepted',test_access=False,files=files)
    atomic_write_json(result,out/'accepted.json');return result
