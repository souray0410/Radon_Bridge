"""Read-only selected-model energy, gradients and paired-message diagnostics."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from radon_bridge.data.observed_pair import collate_observed
from radon_bridge.evaluation.paired_native import move,evaluate
from radon_bridge.runtime.host_checkpoint import capture_rng,restore_rng,cpu_tree
from radon_bridge.runtime.state import atomic_write_json,file_sha256,stable_hash
from radon_bridge.studies.project_build import build


def tensor_state(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def permutations(dataset,out):
    groups={}
    for i,row in enumerate(dataset.first.rows):groups.setdefault(tuple(row['eyes']),[]).append(i)
    if any(len(g)<2 for g in groups.values()):raise ValueError('Eye-pattern singleton prevents full derangement; explicit diagnosis required')
    rng=np.random.default_rng(743190);result=[]
    for _ in range(20):
        p=np.arange(len(dataset))
        for values in groups.values():
            values=np.asarray(values)
            while True:
                q=rng.permutation(values)
                if np.all(q!=values):break
            p[values]=q
        result.append(p)
    path=Path(out)/'permutations.npz';np.savez(path,participant_ids=np.asarray(dataset.participant_ids),permutations=np.stack(result))
    return result


def paired_messages(model,dev,out,device,paused):
    from expanded.native import metrics
    from radon_bridge.analysis.communication import block_messages
    ex=model.task.modules_by_name()['bridge_0_exchange']
    if not hasattr(ex,'channel_bases') or len(ex.keys)!=2:raise ValueError('Paired diagnostic requires fixed channel bridge')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    perms=permutations(dev,out);offsets=np.r_[0,np.cumsum(dev.counts)]
    caches=[np.lib.format.open_memmap(out/f'encoded_{i}.npy',mode='w+',dtype=np.float32,
        shape=(int(offsets[-1]),ex.mixer.widths[i],ex.S)) for i in range(2)]
    reference={'cfp':[],'oct':[]};labels=[]
    loader=lambda:DataLoader(dev,batch_size=1,shuffle=False,collate_fn=collate_observed,num_workers=0,generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for index,batch in enumerate(loader()):
            if paused():raise InterruptedError('Pause during paired cache creation')
            z,_=model(move(batch,device));labels.extend(batch['label'].tolist())
            for branch in reference:reference[branch].append(z[branch].softmax(1).cpu().numpy())
            for i,x in enumerate(ex.latest_inputs):
                caches[i][offsets[index]:offsets[index+1]]=ex.projectors[i](ex.channel_bases[i].encode(x)).cpu().numpy()
        for cache in caches:cache.flush()
        reference={k:np.concatenate(v) for k,v in reference.items()};labels=np.asarray(labels)
        schedule=[('paired',None),('disable_oct_to_cfp',None),('disable_cfp_to_oct',None),('disable_both',None)]
        schedule += [(kind,i) for kind in ('shuffle_oct_to_cfp','shuffle_cfp_to_oct','shuffle_both') for i in range(20)]
        rows=[]
        for kind,repeat in schedule:
            pred={'cfp':[],'oct':[]}
            for index,batch in enumerate(loader()):
                if paused():raise InterruptedError('Pause during read-only pairing diagnostic')
                overrides={}
                directions=[]
                if 'oct_to_cfp' in kind or kind.endswith('both'):directions.append((0,1))
                if 'cfp_to_oct' in kind or kind.endswith('both'):directions.append((1,0))
                for dst,src in directions:
                    if kind.startswith('disable'):overrides[(dst,src)]=None
                    else:
                        permutation=perms[repeat] if dst==0 else np.argsort(perms[repeat]);donor=permutation[index]
                        overrides[(dst,src)]=torch.from_numpy(np.array(caches[src][offsets[donor]:offsets[donor+1]])).to(device)
                hook=ex.mixer.register_forward_hook(lambda module,inputs,output: block_messages(module,inputs,overrides)) if overrides else None
                try:z,_=model(move(batch,device))
                finally:
                    if hook:hook.remove()
                for branch in pred:pred[branch].append(z[branch].softmax(1).cpu().numpy())
            pred={k:np.concatenate(v) for k,v in pred.items()}
            if kind=='paired' and any(not np.allclose(pred[k],reference[k],atol=1e-5,rtol=1e-4) for k in pred):raise ValueError('Full MHD cache replay changed predictions')
            values={k:metrics(labels,p) for k,p in pred.items()}
            delta={k:float(np.mean(abs(pred[k]-reference[k]))) for k in pred}
            path=out/(kind+('_'+str(repeat) if repeat is not None else '')+'.npz')
            np.savez(path,participant_ids=np.asarray(dev.participant_ids),labels=labels,**pred)
            rows.append(dict(condition=kind,repeat=repeat,metrics=values,mean_absolute_probability_change=delta,prediction_sha256=file_sha256(path)))
    result=dict(rows=rows,permutations=20,permutation_sha256=file_sha256(out/'permutations.npz'),
        restriction='label-independent derangements within identical observed-eye patterns; paired eyes move together',
        interpretation='functional dependence at fixed weights, not retraining gain or causal proof',test_access=False)
    atomic_write_json(result,out/'summary.json');return result


def diagnose(spec,root,parents,shapes,bases,train,dev,device,paused,*,arm_ids=None):
    root=Path(root);out=root/'diagnostics';out.mkdir(exist_ok=True)
    selected=spec['arms'] if arm_ids is None else [a for a in spec['arms'] if a['id'] in arm_ids]
    if not selected or (arm_ids is not None and {a['id'] for a in selected}!=set(arm_ids)):
        raise ValueError('Unknown diagnostic scope')
    names=[a['id'] for a in selected]
    receipt=out/('accepted.json' if arm_ids is None else 'accepted_'+stable_hash(names)+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['identity']!=stable_hash(spec):raise ValueError('Diagnostic provenance changed')
        if arm_ids is not None and r.get('arm_ids')!=names:raise ValueError('Diagnostic scope changed')
        for n,s in r['files'].items():
            if file_sha256(out/n)!=s:raise ValueError('Diagnostic evidence changed')
        return r
    order=sorted(range(len(train)),key=lambda i:train.participant_ids[i])[:128]
    atomic_write_json({'ids':[train.participant_ids[i] for i in order],'split':'train'},out/'probe.json')
    records=[]
    for arm in selected:
        name=arm['id'];path=out/(name+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['checkpoint_sha256']!=file_sha256(root/'arms'/name/'best.pt'):
                raise ValueError('Diagnostic checkpoint changed')
            records.append(cached);continue
        active=bases
        if arm.get('host'):active=json.loads((root/'host_bases'/arm['host']/'accepted.json').read_text())['bases']
        model=build(parents,shapes,arm,active,spec['seed'],device)
        model.load_state_dict(torch.load(root/'arms'/name/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
        model.eval();before=tensor_state(model);rng=capture_rng()
        exchanges={k:m for k,m in model.task.modules_by_name().items() if k.endswith('_exchange')}
        sums={k:[0.,0.,0.,0.] for k in exchanges};energies={k:[0.,0.,0.,0.] for k in exchanges}
        loader=DataLoader(train,batch_size=1,shuffle=False,collate_fn=collate_observed,num_workers=0,generator=torch.Generator().manual_seed(0))
        try:
            with torch.no_grad():
                for idx,batch in enumerate(loader):
                    if paused():raise InterruptedError('Pause full training residual diagnostic')
                    model(move(batch,device))
                    for key,ex in exchanges.items():
                        for source,(x,d) in enumerate(zip(ex.latest_inputs,ex.latest_deltas)):
                            xnorm=float(x.double().square().sum());dnorm=float(d.double().square().sum())
                            sums[key][2*source]+=xnorm;sums[key][2*source+1]+=dnorm
                            if hasattr(ex,'channel_bases'):
                                q=ex.channel_bases[source];znorm=float(q.encode(x).double().square().sum())
                                energies[key][2*source]+=xnorm;energies[key][2*source+1]+=znorm
                        if idx==order[0] and hasattr(ex,'channel_bases') and ex.mode=='radon':
                            for source,x in enumerate(ex.latest_inputs):
                                p=ex.projectors[source](ex.channel_bases[source].encode(x))
                                image=p.reshape(len(x),-1,ex.projectors[source].M,ex.S).square().mean((0,1)).sqrt().cpu().numpy()
                                np.savez(out/f'{name}_{key}_{source}_sinogram.npz',projection_rms=image,
                                    directions=np.asarray(ex.projectors[source].metadata['directions']),support=ex.projectors[source].metadata['support'])
            # Two CE gradient probes, participant-weighted and accumulated before cosine.
            grads=[];modules=model.task.modules_by_name();loss=modules['task_loss_sum']
            names=[n for n in modules if n.startswith('bridge_') or n.startswith('cfp_stage3') or n.startswith('oct_stage3')]
            for target in (0,1):
                loss.weights=tuple(float(i==target) for i in range(2));model.zero_grad(set_to_none=True)
                for i in order:
                    if paused():raise InterruptedError('Pause gradient probe')
                    batch=collate_observed([train[i]]);model(move(batch,device),1/len(order));model.backward()
                grads.append({name:torch.cat([p.grad.detach().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for p in modules[name].parameters()]) for name in names if sum(p.numel() for p in modules[name].parameters())})
            loss.weights=(1.,1.);model.zero_grad(set_to_none=True)
            gradient={}
            for key,a in grads[0].items():
                b=grads[1][key];an=float(a.norm());bn=float(b.norm())
                gradient[key]=dict(cfp_norm=an,oct_norm=bn,cosine=float(torch.dot(a,b)/(an*bn)) if an and bn else None,undefined_reason=None if an and bn else 'zero_gradient_norm')
            ratios={key:{'cfp':(v[1]/v[0])**.5 if v[0] else None,'oct':(v[3]/v[2])**.5 if v[2] else None} for key,v in sums.items()}
            retained={key:{'cfp':v[1]/v[0] if v[0] else None,'oct':v[3]/v[2] if v[2] else None} for key,v in energies.items()}
            if arm.get('host') and arm['addition']!='continue':
                host_states={}
                for state,disabled in [('both_on',()),('host_only',(1,)),('new_only',(0,)),('both_off',(0,1))]:
                    hooks=[modules[f'bridge_{i}_exchange'].register_forward_hook(
                        lambda module,inputs,output:torch.cat([x.flatten(1) for x in inputs],1)) for i in disabled]
                    try:
                        host_states[state]=evaluate(model,DataLoader(dev,batch_size=1,shuffle=False,collate_fn=collate_observed,num_workers=0,generator=torch.Generator().manual_seed(0)),device,out/(name+'_'+state+'.npz'),paused)
                    finally:
                        for hook in hooks:hook.remove()
                atomic_write_json(dict(states=host_states,interpretation='fixed-model functional deletion, not training gain',test_access=False),out/(name+'_host_states.json'))
            if name in ('svd_radon','qr_radon'):

                paired_messages(model,dev,out/(name+'_pairing'),device,paused)
            row=dict(arm=name,train_participants=len(train),probe_participants=len(order),residual_ratio=ratios,
                retained_energy=retained,gradients=gradient,checkpoint_sha256=file_sha256(root/'arms'/name/'best.pt'),
                energy_scope='selected-model training features before writeback; initial basis energy is separately stored in basis metadata',test_access=False)
            if tensor_state(model)!=before:raise ValueError('Read-only diagnosis changed parameters or BN')
            atomic_write_json(row,path);records.append(row)
        finally:
            model.zero_grad(set_to_none=True);restore_rng(rng);del model;gc.collect()
    render_sinograms(out)
    files={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*')
           if p.is_file() and not p.name.startswith('accepted') and
           (arm_ids is None or p.name in ('probe.json','FIGURES.zh-CN.md') or
            any(str(p.relative_to(out)).startswith(n+'.') or str(p.relative_to(out)).startswith(n+'_') for n in names))}
    result=dict(identity=stable_hash(spec),arm_ids=names,records=records,files=files,state='accepted',test_access=False)
    atomic_write_json(result,receipt);return result


def render_sinograms(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output)
    for path in out.glob('*_sinogram.npz'):
        with np.load(path,allow_pickle=False) as z:
            projection=z['projection_rms'];support=z['support'];directions=z['directions']
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        image=ax.imshow(projection,origin='lower',aspect='auto',extent=[*support,0,len(directions)],cmap='viridis')
        ax.set(xlabel='Normalized projection coordinate s',ylabel='Direction index (EEM order)',title='Radon_Bridge | projected feature RMS')
        fig.colorbar(image,ax=ax,label='RMS across retained channels and observed eyes')
        fig.savefig(path.with_suffix('.png'),dpi=180);fig.savefig(path.with_suffix('.svg'));plt.close(fig)
    (out/'FIGURES.zh-CN.md').write_text('正弦图横轴为归一化投影位置s，纵轴为EEM方向索引，颜色是保留通道及有效眼的投影RMS。3D方向位于球面，不能把纵轴解释为单一平面角度。选取固定排序第一位训练参与者；亮度不代表诊断准确性或临床重要性。此图描述特征，不能证明物理配准。\n')
