"""Resumable full-training sufficient statistics from frozen selected native graphs."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from radon_bridge.data.observed_pair import collate_observed
from radon_bridge.evaluation.paired_native import move
from radon_bridge.methods.basis import save_basis,save_centered_basis,save_random_basis
from radon_bridge.runtime.host_checkpoint import atomic_save,cpu_tree
from radon_bridge.runtime.state import atomic_write_json,file_sha256


def fit(model,dataset,stages,out,identity,seed,device,should_pause=lambda:False):
    if dataset.split!='train' or dataset.augment:raise ValueError('Basis requires unchanged training-only data')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'accepted.json').exists():
        receipt=json.loads((out/'accepted.json').read_text())
        if receipt['identity']!=identity:raise ValueError('Basis identity changed')
        for bases in receipt['bases'].values():
            for ref in bases.values():
                if file_sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Basis artifact changed')
        return receipt['bases']
    names=[f'{source}_stage{stage}' for stage in stages for source in ('cfp','oct')]
    offset=0;moments={};sums={};counts={}
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['names']!=names:raise ValueError('Basis resume provenance changed')
        offset=state['offset'];moments=state['moments'];sums=state['sums'];counts=state['counts']
    before=cpu_tree(model.state_dict());mode=model.training;model.eval()
    def checkpoint():atomic_save(out/'last.pt',dict(identity=identity,names=names,offset=offset,moments=moments,sums=sums,counts=counts))
    try:
        loader=DataLoader(Subset(dataset,range(offset,len(dataset))),batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_observed,generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            for batch in loader:
                if should_pause():checkpoint();raise InterruptedError('Pause basis at participant boundary')
                model(move(batch,device))
                prewrite={key:x for m in model.task.modules_by_name().values()
                    if hasattr(m,'keys') and getattr(m,'latest_inputs',None) is not None
                    for key,x in zip(m.keys,m.latest_inputs)}
                for name in names:
                    x=prewrite[name] if name in prewrite else model.task.by_name[name].feature_message.current_state
                    c=x.shape[1];v=x.movedim(1,-1).reshape(-1,c)
                    if name not in moments:moments[name]=torch.zeros(c,c,dtype=torch.float64);sums[name]=torch.zeros(c,dtype=torch.float64);counts[name]=0
                    for block in v.split(4096):
                        block=block.double();moments[name]+=(block.T@block).cpu();sums[name]+=block.sum(0).cpu();counts[name]+=len(block)
                offset+=len(batch['label'])
                if offset%128==0:checkpoint()
        checkpoint()
        if any(not torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items()):raise ValueError('Basis fitting modified native state')
        bases={kind:{} for kind in ('fixed_svd_channel','fixed_centered_svd_channel','fixed_random_orthogonal_channel')}
        for name in names:
            shard=out/(name+'_receipt.json')
            if shard.exists():
                row=json.loads(shard.read_text())
                if row['identity']!=identity:raise ValueError('Basis shard changed')
            else:
                # Only completed receipts are reusable. Interrupted writer paths
                # remain attempts, separate from the new content publication.
                import time
                dest=out/name/str(time.time_ns())
                provenance={'parent_identity':identity,'participants':len(dataset),'source_node':name}
                svd=save_basis(moments[name],counts[name],dest,name,seed,provenance)
                centered=save_centered_basis(moments[name],sums[name],counts[name],dest,name,seed,provenance)
                qr=save_random_basis(svd,dest)
                row=dict(identity=identity,refs=dict(zip(bases,(svd,centered,qr))));atomic_write_json(row,shard)
            for kind,ref in row['refs'].items():bases[kind][name]=ref
        atomic_write_json(dict(identity=identity,bases=bases,fit_split='train',participants=len(dataset),test_access=False),out/'accepted.json')
        return bases
    finally:model.train(mode)
