"""Participant-preserving perturbations of one fixed linear communication stage."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from radon_bridge.analysis.communication import block_messages
from radon_bridge.evaluation.paired_native import move
from radon_bridge.runtime.state import atomic_write_json, file_sha256


def permutations(dataset):
    key=dataset.sources[0]['key'];ds=dataset.datasets[key];groups={}
    for i,pid in enumerate(dataset.participant_ids):
        row=ds.rows[dataset.indices[key][pid]]
        groups.setdefault(tuple(row['eyes']),[]).append(i)
    if any(len(v)<2 for v in groups.values()):raise ValueError('Eye-pattern singleton prevents derangement')
    rng=np.random.default_rng(743190);result=[]
    for _ in range(20):
        p=np.arange(len(dataset))
        for values in groups.values():
            values=np.asarray(values)
            while True:
                draw=rng.permutation(values)
                if np.all(draw!=values):break
            p[values]=draw
        result.append(p)
    return np.stack(result)


def _diagnose(model, dataset, output, device, paused):
    from radon_bridge.evaluation.metrics import binary_metrics as metrics
    if dataset.split!='development' or dataset.augment:raise ValueError('Unchanged development view required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    modules=model.task.modules_by_name();exchanges=[m for n,m in modules.items() if n.endswith('_exchange')]
    if len(exchanges)!=1:raise ValueError('This fixed pairing protocol requires a single bridge')
    ex=getattr(exchanges[0],'exchange',exchanges[0])
    if ex.compression == 'factorized_projected':
        if ex.mode not in ('radon','linear_resample'):
            raise ValueError('Unsupported factorized geometry')
        def encode(i,x): return ex.projectors[i](x)
    elif hasattr(ex,'channel_bases'):
        def encode(i,x): return ex.projectors[i](ex.channel_bases[i].encode(x))
    else:
        raise ValueError('Pairing requires fixed-channel or global factorized communication')
    keys=model.source_keys;nsource=len(keys);offset=np.r_[0,np.cumsum(dataset.counts)]
    perms=permutations(dataset);inverses=np.argsort(perms,axis=1)
    np.savez(out/'permutations.npz',participant_ids=np.asarray(dataset.participant_ids),permutations=perms)
    loader=lambda:DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=dataset.collate_fn,num_workers=0,
                             generator=torch.Generator().manual_seed(0))
    model.eval();reference={k:[] for k in keys};labels={k:[] for k in keys}
    # Encoded tensors are reproducible scratch, not accepted result dependencies.
    # TemporaryDirectory limits disk use to one checkpoint; predictions are retained.
    with tempfile.TemporaryDirectory(prefix='pairing_scratch_',dir=out) as scratch, torch.no_grad():
        caches=[np.lib.format.open_memmap(Path(scratch)/f'{i}.npy',mode='w+',dtype=np.float32,
            shape=(int(offset[-1]),ex.mixer.widths[i],ex.S)) for i in range(nsource)]
        for index,batch in enumerate(loader()):
            if paused():raise InterruptedError('Pause fixed pairing cache')
            z,_=model(move(batch,device))
            for k in keys:reference[k].append(z[k].softmax(1).cpu().numpy());labels[k].extend(batch['labels'][k].tolist())
            for i,x in enumerate(ex.latest_inputs):
                caches[i][offset[index]:offset[index+1]]=encode(i,x).cpu().numpy()
        for cache in caches:cache.flush()
        reference={k:np.concatenate(v) for k,v in reference.items()};labels={k:np.asarray(v) for k,v in labels.items()}
        selectors=[(f'sender_{i}',[i]) for i in range(nsource)]+[('all',list(range(nsource)))]
        schedule=[('paired',[],None)]
        schedule += [('disable_'+name,senders,None) for name,senders in selectors]
        schedule += [('shuffle_'+name,senders,repeat) for name,senders in selectors for repeat in range(20)]
        rows=[]
        for condition,senders,repeat in schedule:
            predictions={k:[] for k in keys}
            for index,batch in enumerate(loader()):
                if paused():raise InterruptedError('Pause pairing perturbation')
                overrides={}
                if condition=='paired':
                    for src in range(nsource):
                        for dst in range(nsource):
                            if src!=dst:
                                overrides[(dst,src)]=torch.from_numpy(np.array(caches[src][offset[index]:offset[index+1]])).to(device)
                for src in senders:
                    for dst in range(nsource):
                        if dst==src:continue
                        if condition.startswith('disable'):overrides[(dst,src)]=None
                        else:
                            # Opposite directed edges use inverse permutations. For
                            # six sources this is edge-pair consistency, not a single
                            # global six-person reassembly, and is reported as such.
                            p=perms[repeat] if dst<src else inverses[repeat];donor=p[index]
                            overrides[(dst,src)]=torch.from_numpy(np.array(caches[src][offset[donor]:offset[donor+1]])).to(device)
                hook=ex.mixer.register_forward_hook(lambda module,inputs,output:block_messages(module,inputs,overrides)) if overrides else None
                try:z,_=model(move(batch,device))
                finally:
                    if hook:hook.remove()
                for k in keys:predictions[k].append(z[k].softmax(1).cpu().numpy())
            predictions={k:np.concatenate(v) for k,v in predictions.items()}
            if condition=='paired' and any(not np.allclose(reference[k],predictions[k],atol=1e-5,rtol=1e-4) for k in keys):
                raise ValueError('Full MHD pairing replay mismatch')
            path=out/(condition+('' if repeat is None else '_'+str(repeat))+'.npz')
            np.savez_compressed(path,participant_ids=np.asarray(dataset.participant_ids),**predictions,
                                **{'labels__'+k:v for k,v in labels.items()})
            row=dict(condition=condition,repeat=repeat,metrics={k:metrics(labels[k],predictions[k]) for k in keys},
                     mean_absolute_probability_change={k:float(abs(predictions[k]-reference[k]).mean()) for k in keys},
                     predictions_sha256=file_sha256(path))
            rows.append(row);atomic_write_json(dict(completed=len(rows),total=len(schedule),test_access=False),out/'status.json')
        aggregates={}
        for name,_ in selectors:
            subset=[r for r in rows if r['condition']=='shuffle_'+name]
            aggregates[name]={}
            for key in keys:
                values=np.asarray([r['metrics'][key]['macro_f1'] for r in subset])
                aggregates[name][key]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),minimum=float(values.min()),maximum=float(values.max()))
    result=dict(rows=rows,permutation_aggregates=aggregates,permutations_sha256=file_sha256(out/'permutations.npz'),
        conditions=len(schedule),test_access=False,paired_eyes_preserved=True,
        compression=ex.compression, source_edge_definition='B_dst K A_src' if ex.compression=='factorized_projected' else 'dense source block',
        interpretation='fixed-model functional dependence; reciprocal edge-pair derangements; not clinical causality or global six-person reassignment')
    atomic_write_json(result,out/'summary.json');return result


def diagnose(model, dataset, output, device, paused):
    """Read-only diagnostics; callers must separately admit the real cache workload."""
    from radon_bridge.runtime.host_checkpoint import capture_rng, restore_rng
    from radon_bridge.analysis.native_diagnostics import tensor_state
    modes=[(m,m.training) for m in model.modules()]
    before=tensor_state(model);rng=capture_rng()
    try:
        result=_diagnose(model,dataset,output,device,paused)
        if tensor_state(model)!=before:raise ValueError('Pairing altered weights or BN')
        return result
    finally:
        for module,training in modes:module.training=training
        restore_rng(rng)
