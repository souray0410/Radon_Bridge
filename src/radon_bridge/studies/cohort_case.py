"""Finite small-cohort development study, separate from Ibex large-cohort work.

Resume from the last atomic epoch boundary, replaying an interrupted partial
epoch with its original RNG and sample order. Never treat a cap as completion.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from radon_bridge.runtime.pilot_checkpoint import read as read_state, save as save_state
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.models.model import PilotGraph
from radon_bridge.training.optimization import configure_optimizer, clip_task_gradients
from radon_bridge.training.convergence import Plateau
from radon_bridge.training.trainer import loader, evaluate, write_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def rng():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(r):
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state(r['cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def build(cfg):
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    g=PilotGraph(bridge_configs=cfg['bridges'],seed=cfg['seed'],device='cuda')
    for branch,ref in cfg['parents'].items():
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Parent SHA mismatch')
        p=read_state(ref['path'],kind='native_parent')
        if p.get('branch')!=branch or p.get('seed')!=cfg['seed'] or p.get('stop_reason')!='validation_plateau':
            raise ValueError('Unaccepted parent')
        g.load_native_state(p['model'],branch)
    if 'augmentation_host' in cfg:
        from radon_bridge.studies.cohort_augmentation import load_host
        load_host(cfg,g)
    opt=configure_optimizer(g,dict(adapt_stages=[1,2,3,4],backbone_lr=6e-5,
        head_lr=1e-4,bridge_lr=1e-4,weight_decay=.01,training_regime='full_finetune'))
    return g,opt


def check_resources():
    import psutil
    if psutil.virtual_memory().available < .15*psutil.virtual_memory().total:
        raise MemoryError('Host reserve below 15 percent')
    return False


def update(g,opt,batch):
    check_resources()
    g.graph.train();opt.zero_grad(set_to_none=True)
    c,o,y=batch[:3];_,loss=g.forward(c.cuda(),o.cuda(),y.cuda());g.backward()
    clip_task_gradients(g,5.);opt.step();return float(loss.detach())


def datasets(data):
    train=PairedDataset(data,'train',224);dev=PairedDataset(data,'validation',224)
    if len(train)!=1264 or len(dev)!=296:raise ValueError('Small-cohort role mismatch')
    if {r['participant_id'] for r in train.rows}&{r['participant_id'] for r in dev.rows}:raise ValueError('Split overlap')
    return train,dev


def configure_device():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # Historical parents used the cuDNN default TF32 convolution policy.
    # Match it explicitly; changing it breaks exact saved-prediction replay.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    # Configure from actual device occupancy; no platform-specific reserve.
    from mhd_models.scheduling.gpu_budget import configure_allocator
    configure_allocator(0)


def grouped_structure_probe(g,cfg):
    grouped=[(i,c) for i,c in enumerate(cfg.get('bridges',[])) if c.get('group_count',1)>1]
    if not grouped:return None
    if len(grouped)!=1:raise ValueError('WS02 grouped supplement expects one grouped bridge')
    index,config=grouped[0];groups=config['group_count']
    mixer=g.modules_by_name()[f'bridge_{index}_exchange'].mixer
    if mixer.conv.groups!=groups or not hasattr(mixer,'group_permutation') or not hasattr(mixer,'group_inverse'):
        raise ValueError('Grouped mixer structure mismatch')
    permutation=mixer.group_permutation.detach().cpu();inverse=mixer.group_inverse.detach().cpu()
    if not torch.equal(permutation.index_select(0,inverse),torch.arange(len(permutation))):
        raise ValueError('Grouped inverse permutation mismatch')
    widths=list(mixer.widths)
    if len(widths)!=2 or sum(widths)%groups:raise ValueError('Grouped WS02 source widths invalid')
    offsets=[0,widths[0],sum(widths)];per_group=sum(widths)//groups;counts=[]
    for group in range(groups):
        segment=permutation[group*per_group:(group+1)*per_group]
        row=[]
        for start,end in zip(offsets[:-1],offsets[1:]):row.append(int(((segment>=start)&(segment<end)).sum()))
        if min(row)<=0:raise ValueError('Every group must contain every source')
        counts.append(row)
    probe=copy.deepcopy(mixer).cpu()
    with torch.no_grad():probe.conv.weight.fill_(1)
    features=[torch.ones(1,width,5,requires_grad=True) for width in widths]
    output=probe(*features)
    first_group=permutation[:per_group];second_group=permutation[per_group:2*per_group]
    dst=int(first_group[first_group<offsets[1]][0])
    within=int(first_group[first_group>=offsets[1]][0])-offsets[1]
    cross=int(second_group[second_group>=offsets[1]][0])-offsets[1]
    gradient=torch.autograd.grad(output[0][:,dst].sum(),features[1])[0]
    within_value=float(gradient[0,within].abs().sum())
    cross_value=float(gradient[0,cross].abs().sum())
    if not within_value>0 or cross_value!=0:raise ValueError('Grouped direct-gradient structure mismatch')
    return dict(group_count=groups,conv_groups=mixer.conv.groups,source_widths=widths,group_source_counts=counts,
        within_group_cross_source_gradient=within_value,cross_group_direct_gradient=cross_value,
        inverse_permutation_verified=True)


def profile(cfg,data,out):
    out.mkdir(parents=True,exist_ok=False);configure_device();train,dev=datasets(data)
    g,opt=build(cfg);b=next(iter(loader(train,16,cfg['seed'],0)))
    ids={n:g.by_name[n].id for n in ('cfp_stage3','oct_stage3')}
    g.graph.eval()
    with torch.no_grad(): initial,_=g.forward(b[0].cuda(),b[1].cuda(),b[2].cuda());initial={k:v.cpu() for k,v in initial.items()}
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    physical=[]
    def sample_physical():
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        values=[float(line.split(',')[1].strip())/1024 for line in text.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
        if values:physical.append(max(values))
    for _ in range(5+20):
        update(g,opt,b);sample_physical()
    g.graph.eval()
    with torch.no_grad():
        _,loss=g.forward(b[0].cuda(),b[1].cuda(),b[2].cuda())
        _,direct_loss=g.native_forward(b[0].cuda(),b[1].cuda(),b[2].cuda())
        torch.testing.assert_close(loss,direct_loss,atol=1e-6,rtol=1e-5)
    g.graph.train();opt.zero_grad(set_to_none=True);g.forward(b[0].cuda(),b[1].cuda(),b[2].cuda());g.backward()
    actual=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in g.graph.parameters()]
    for node in g.graph.nodes:node.reset()
    opt.zero_grad(set_to_none=True);_,loss=g.native_forward(b[0].cuda(),b[1].cuda(),b[2].cuda());loss.backward()
    for before,p in zip(actual,g.graph.parameters()):
        if before is not None:torch.testing.assert_close(before,p.grad.cpu(),atol=3e-5,rtol=3e-4)
    if cfg['bridges']:
        g.graph.eval();g.forward(b[0][:2].cuda(),b[1][:2].cuda(),b[2][:2].cuda())
        for dst,src in [('cfp','oct'),('oct','cfp')]:
            grad=torch.autograd.grad(g.by_name[dst+'_loss'].feature_message.current_state,
                next(g.modules_by_name()[src+'_stage1'].parameters()),retain_graph=True,allow_unused=True)[0]
            if grad is None:grad=torch.zeros(1)
            assert torch.isfinite(grad).all()
            if all(x.get('family','radon')=='radon' and x.get('mode')=='self' for x in cfg['bridges']):
                assert grad.abs().sum()==0
            else:assert grad.abs().sum()>0
    snapshot=dict(model=g.save_state(),optimizer=copy.deepcopy(opt.state_dict()),rng=rng())
    save_state(snapshot,out/'resume.pt',kind='profile_resume');update(g,opt,b)
    expected=g.save_state();expected_opt=copy.deepcopy(opt.state_dict())
    saved=read_state(out/'resume.pt',kind='profile_resume')
    g.load_complete_state(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);update(g,opt,b)
    assert same(expected,g.save_state()) and same(expected_opt,opt.state_dict())
    assert ids=={n:g.by_name[n].id for n in ids}
    val=evaluate(g,dev,16,cfg['seed'],stop=check_resources)
    grouped_structure=grouped_structure_probe(g,cfg)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/1024**3
    sample_physical()
    receipt=dict(passed=True,scope='resource_and_MHD_development_only',test_used=False,
        formal_updates=0,seed=cfg['seed'],configuration=cfg,warmup=5,measured_updates=20,
        checkpoint_update_exact=True,node_ids_preserved=True,autograd_equivalence=True,
        full_development_participants=296,peak_reserved_gib=peak,peak_process_sampled_gib=max(physical) if physical else None,seconds=time.monotonic()-tick,
        bridge_parameters=sum(p.numel() for n,m in g.modules_by_name().items() if n.startswith('bridge_') for p in m.parameters()),
        torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
        numerical_policy=dict(parameter_dtype="float32",autocast=False,matmul_tf32=False,cudnn_tf32=True),
        resume_sha256=sha(out/'resume.pt'))
    if grouped_structure is not None:receipt['grouped_structure']=grouped_structure
    write_json(out/'accepted.json',receipt)


def train_case(cfg,data,out):
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'accepted.json').exists():raise RuntimeError('Do not duplicate an accepted run')
    configure_device();train,dev=datasets(data);g,opt=build(cfg)
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    progress=lambda **kw:write_json(out/'status.json',dict(pid=os.getpid(),updated_at=time.time(),test_used=False,**kw))
    policy=dict(min_epochs=8,max_epochs=60,patience=6,min_delta=.001,lr_patience=3,lr_factor=.3)
    monitor=Plateau(**policy);history=[];epoch=0
    identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    def checkpoint():
        save_state(dict(identity=identity,model=g.save_state(),optimizer=opt.state_dict(),rng=rng(),
            epoch=epoch,monitor=vars(monitor),history=history,configuration=cfg,
            selected=read_state(out/'best.pt',kind='selected',configuration=cfg)),out/'resume.pt',kind='resume')
    if (out/'resume.pt').exists():
        p=read_state(out/'resume.pt',kind='resume',configuration=cfg)
        if p['identity']!=identity:raise ValueError('Resume identity changed')
        g.load_complete_state(p['model']);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'])
        epoch=p['epoch'];monitor.__dict__.update(p['monitor']);history=p['history']
        atomic_save(p['selected'],out/'best.pt')
    else:
        initial=evaluate(g,dev,16,cfg['seed'],out/'initial_predictions.npz',stop=check_resources)
        with np.load(out/'initial_predictions.npz',allow_pickle=False) as current:
            if 'augmentation_host' in cfg:
                manifest=json.loads(Path(cfg['augmentation_host']['path']).read_text())
                references={k:Path(manifest['predictions']['path']) for k in ('cfp','oct')}
            else:
                references={k:Path(v['path']).parent/'selected_predictions.npz' for k,v in cfg['parents'].items()}
            for key,path in references.items():
                with np.load(path,allow_pickle=False) as previous:
                    assert np.array_equal(current['ids'],previous['ids']) and np.array_equal(current['y'],previous['y'])
                    np.testing.assert_allclose(current[key],previous[key],rtol=1e-5,atol=1e-6)
        monitor.update(initial['mean_task_macro_f1'],0)
        save_state(dict(model=g.save_state(),epoch=0,metrics=initial,configuration=cfg),out/'best.pt',kind='selected');checkpoint()
        write_json(out/'initial_acceptance.json',dict(strict_parent_predictions=True,test_used=False,
            numerical_policy=dict(parameter_dtype="float32",autocast=False,matmul_tf32=False,cudnn_tf32=True)))
    converged=epoch>=8 and monitor.bad>=6
    try:
        for current_epoch in range(epoch+1,61):
            if converged:break
            tick=time.monotonic();seen=0
            for b in loader(train,16,cfg['seed'],current_epoch-1):
                if stop[0]:raise InterruptedError('Replay partial epoch from last accepted boundary')
                loss=update(g,opt,b);seen+=len(b[2])
                if seen%128==0:progress(state='training',epoch=current_epoch,participants=seen,loss=loss)
            scores=evaluate(g,dev,16,cfg['seed'],stop=lambda:stop[0] or check_resources())
            flags=monitor.update(scores['mean_task_macro_f1'],current_epoch)
            if flags['improved']:save_state(dict(model=g.save_state(),epoch=current_epoch,metrics=scores,configuration=cfg),out/'best.pt',kind='selected')
            if flags['reduce_lr']:
                for group in opt.param_groups:group['lr']*=.3
            epoch=current_epoch;converged=flags['plateau']
            history.append(dict(epoch=epoch,metrics=scores,seconds=time.monotonic()-tick,flags=flags,monitor=monitor.state()))
            checkpoint();write_json(out/'history.json',history);progress(state='training',epoch=epoch,metrics=scores)
        if not converged:
            progress(state='needs_review',reason='epoch_cap_without_plateau',epoch=epoch);return 2
        selected=read_state(out/'best.pt',kind='selected',configuration=cfg);g.load_complete_state(selected['model'])
        scores=evaluate(g,dev,16,cfg['seed'],out/'selected_predictions.npz',stop=lambda:stop[0] or check_resources())
        files=['best.pt','resume.pt','selected_predictions.npz','history.json','initial_acceptance.json']
        write_json(out/'accepted.json',dict(identity=identity,state='complete',converged_by_policy=True,
            stop_reason='validation_plateau',epochs_ran=epoch,best_epoch=selected['epoch'],configuration=cfg,
            selected_validation=scores,test_used=False,files={n:sha(out/n) for n in files}))
        progress(state='complete',epoch=epoch,metrics=scores);return 0
    except InterruptedError:
        progress(state='paused',resume_epoch=epoch,partial_epoch_replayed=True);return 75


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['profile','train']);p.add_argument('--config',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    from radon_bridge.runtime.exclusive_gpu import acquire
    device_lock=acquire()
    try:
        if a.mode=='profile':profile(cfg,a.data,Path(a.output))
        else:sys.exit(train_case(cfg,a.data,Path(a.output)))
    except Exception:
        import traceback
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(time=time.time(),traceback=traceback.format_exc(),test_used=False));raise


if __name__=='__main__':main()
