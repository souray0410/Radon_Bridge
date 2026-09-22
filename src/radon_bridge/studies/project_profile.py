"""Actual complete-graph update, validation, checkpoint/reload and resource admission."""
import copy
import os
from pathlib import Path
import time
import torch
from radon_bridge.evaluation.paired_native import move
from radon_bridge.runtime.host_checkpoint import capture_rng,restore_rng,cpu_tree,save,load
from radon_bridge.runtime.state import atomic_write_json,file_sha256
from radon_bridge.training.paired_native import Schedule


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def profile_model(model,batch,cfg,identity,out,device,*,production=True):
    import psutil
    out=Path(out);out.mkdir(parents=True,exist_ok=True);saved=cpu_tree(model.state_dict());rng=capture_rng();training=model.training
    if production and device.type!='cuda':raise ValueError('Production admission requires actual CUDA execution')
    process=psutil.Process();rss=process.memory_info().rss;start=time.time();samples=[]
    if device.type=='cuda':
        total=torch.cuda.get_device_properties(device).total_memory;budget=total
        torch.cuda.reset_peak_memory_stats(device)
    else:total=budget=0
    opt=torch.optim.AdamW(model.groups(cfg['backbone_lr'],cfg['head_lr'],cfg['bridge_lr']),weight_decay=cfg['weight_decay'])
    sch=Schedule(opt,cfg);opt.zero_grad(set_to_none=True);nodes=model.node_identity()
    def update():
        nonlocal rss
        model.train()
        for _ in range(cfg['effective_batch']//len(batch['label'])):
            _,loss=model(move(batch,device),len(batch['label'])/cfg['effective_batch']);model.backward()
            if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
        model.clip(cfg['clip']);opt.step();opt.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        rss=max(rss,process.memory_info().rss)
        samples.append(dict(allocated=torch.cuda.max_memory_allocated(device) if total else 0,
            reserved=torch.cuda.max_memory_reserved(device) if total else 0,
            device_used=total-torch.cuda.mem_get_info(device)[0] if total else 0))
    try:
        for _ in range(5):update()
        for _ in range(20):update()
        save(out/'resume.pt',model=model,optimizer=opt,scheduler=sch,identity=identity,progress={'updates':25},node_ids=nodes)
        update();expected=cpu_tree(model.state_dict());expected_opt=cpu_tree(opt.state_dict())
        load(out/'resume.pt',model=model,optimizer=opt,scheduler=sch,identity=identity,node_ids=nodes);update()
        if not same(expected,cpu_tree(model.state_dict())) or not same(expected_opt,cpu_tree(opt.state_dict())):raise ValueError('Full next-update recovery differs')
        model.eval()
        with torch.no_grad():
            z,_=model(move(batch,device))
            if any(v.shape!=(len(batch['label']),2) or not torch.isfinite(v).all() for v in z.values()):raise ValueError('Evaluation preflight failed')
        peak=max(s['device_used'] for s in samples)
        if production and peak>budget:raise ValueError('Full-device reserve exceeded')
        # SLURM_MEM_PER_NODE is MiB, distinct from host physical memory. Native
        # cgroup protections remain in force; use the actual allocated envelope.
        import os
        allocated=int(os.environ.get('SLURM_MEM_PER_NODE','0'))*1024**2
        if production and (not allocated or rss>.85*allocated):raise ValueError('Allocated host memory reserve not verified')
        result=dict(status='accepted',case_identity=identity,production_gpu_admission=production,
            exact_next_update=True,initial_state_restored=True,maximum_observed_eyes=max(batch['counts']),
            measured_effective_batch=cfg['effective_batch'],warmups=5,updates=20,peak_device_bytes=peak,
            peak_reserved_bytes=max(s['reserved'] for s in samples),host_rss_bytes=rss,seconds=time.time()-start,test_access=False,
            checkpoint_sha256=file_sha256(out/'resume.pt'))
        atomic_write_json(result,out/'accepted.json');return result
    finally:
        model.load_state_dict(saved,strict=True);model.zero_grad(set_to_none=True);model.train(training);restore_rng(rng)


def profile(spec,output,device):
    # The dispatcher probes the intact largest-parent pair first. Every actual
    # bridge arm has a separate mandatory profile in project_case before training.
    from radon_bridge.studies.project_case import prepare
    from radon_bridge.studies.project_build import build
    from radon_bridge.data.observed_pair import collate_observed
    from radon_bridge.runtime.state import stable_hash
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if device.type=='cuda' and os.environ.get('RADON_PROBE_MAX_BYTES'):
        total=torch.cuda.get_device_properties(device).total_memory
        from mhd_models.scheduling.gpu_budget import configure_allocator
        configure_allocator(device, cap=int(os.environ['RADON_PROBE_MAX_BYTES']))
    parents,shapes,train,fit,dev=prepare(spec,out,device,lambda:False)
    indices=[i for i,n in enumerate(train.counts) if n==2][:spec['training']['microbatch']]
    if len(indices)!=spec['training']['microbatch']:raise ValueError('Maximum-eye fixture unavailable')
    batch=collate_observed([train[i] for i in indices]);model=build(parents,shapes,spec['arms'][0],{},spec['seed'],device)
    return profile_model(model,batch,spec['training'],stable_hash(spec),out,device)
