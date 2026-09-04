"""One isolated, fully trainable paired-network trial. No adaptive decisions here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PairedDataset
from .metrics import classification_metrics
from .model import PilotGraph
from .optimization import configure_optimizer, clip_task_gradients


def write_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)


def bridge_configs(stages, mode='radon'):
    return [{'nodes':[f'cfp_stage{s}',f'oct_stage{s}'],'M':16,'S':64,'H':512,'mode':mode} for s in stages]


def loader(data, batch, seed, epoch=None):
    return DataLoader(data,batch_size=batch,shuffle=epoch is not None,num_workers=2,pin_memory=True,
                      generator=torch.Generator().manual_seed(seed+(epoch or 0)))


@torch.no_grad()
def evaluate(g, data, batch, seed, output=None, stop=None):
    g.graph.eval(); scores={k:[] for k in g.branches}; labels=[]; ids=[]; ratios={}
    for c,o,y,keys in loader(data,batch,seed):
        if stop and stop(): raise InterruptedError('Trial stopped by controller')
        logits,_=g.forward(c.cuda(),o.cuda(),y.cuda())
        for k in scores: scores[k].append(logits[k].softmax(1).cpu().numpy())
        labels.extend(y.tolist()); ids.extend(keys)
        for i, meta in enumerate(g.communication_groups):
            ex=g.modules_by_name()[meta['exchange_edge_name']]
            for k,x,dx in zip(ex.keys,ex.latest_inputs,ex.latest_deltas):
                row=ratios.setdefault(f'{i}_{k}',[0.,0.]); row[0]+=float(dx.square().sum()); row[1]+=float(x.square().sum())
    arrays={k:np.concatenate(v) for k,v in scores.items()}
    if output: np.savez(output, ids=np.asarray(ids), y=np.asarray(labels), **arrays)
    tasks={k:classification_metrics(labels,v) for k,v in arrays.items()}
    return {'tasks':tasks,'mean_task_macro_f1':float(np.mean([v['macro_f1'] for v in tasks.values()])),
            'delta_over_feature_l2':{k:math_sqrt(a/b) if b else 0. for k,(a,b) in ratios.items()}}


def math_sqrt(x): return float(np.sqrt(x))


def parameter_hash(g):
    h=hashlib.sha256()
    for name,module in sorted(g.modules_by_name().items()):
        if name.startswith('bridge_'):continue
        h.update(name.encode())
        for key,p in sorted(module.state_dict().items()):
            h.update(key.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(3); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    cfg=json.loads(Path(args.config).read_text()); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Completed trial must not be overwritten')
    seed=cfg['seed']; np.random.seed(seed); torch.manual_seed(seed)
    stop_requested=False
    def stop_signal(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop_signal)
    train=PairedDataset(args.data,'train',224); val=PairedDataset(args.data,'validation',224)
    assert len(train)==1264 and len(val)==296
    assert not ({r['participant_id'] for r in train.rows}&{r['participant_id'] for r in val.rows})
    recipe={'adapt_stages':[1,2,3,4],'backbone_lr':cfg['backbone_lr'],'head_bridge_lr':1e-4,
            'weight_decay':.01,'training_regime':'full_finetune'}
    g=None; opt=None; epoch=0; start=time.monotonic()
    try:
        g=PilotGraph(bridge_configs=cfg['bridges'],seed=seed,device='cuda')
        opt=configure_optimizer(g,recipe)
        initial_hash=parameter_hash(g)
        info={'configuration':cfg,'initial_native_sha256':initial_hash,'groups':g.communication_groups,
              'parameters':sum(p.numel() for p in g.graph.parameters()),
              'trainable_parameters':sum(p.numel() for p in g.graph.parameters() if p.requires_grad),
              'batchnorm_policy':'train','initialization':'CFP ImageNet; OCT inflated ImageNet, not OCT-specific pretraining'}
        if any(not p.requires_grad for p in g.graph.parameters()): raise AssertionError('Unexpected frozen parameter')
        write_json(out/'model.json',info)
        batch=cfg['microbatch']; torch.cuda.reset_peak_memory_stats()
        def progress(**kw):
            write_json(out/'progress.json',{'pid':os.getpid(),'epoch':epoch,'elapsed_seconds':time.monotonic()-start,**kw})
        if cfg.get('profile'):
            g.graph.train(); c,o,y,_=next(iter(loader(train,batch,seed,0)))
            before={k:[p.detach().clone() for p in m.parameters()] for k,m in g.modules_by_name().items() if list(m.parameters())}
            norm_records=[]; step_times=[]
            for step in range(3):
                if stop_requested: raise InterruptedError('Profile stopped')
                opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); tick=time.monotonic()
                _,loss=g.forward(c.cuda(),o.cuda(),y.cuda()); assert torch.isfinite(loss)
                g.backward()
                norms={k:float(v) for k,v in clip_task_gradients(g,5.).items()}; norm_records.append(norms)
                opt.step(); torch.cuda.synchronize(); step_times.append(time.monotonic()-tick)
                progress(step=step+1,train_loss=float(loss))
            changed={k:any(not torch.equal(a,p) for a,p in zip(old,g.modules_by_name()[k].parameters())) for k,old in before.items()}
            assert all(changed.values()),changed
            # Verify gradients from one task to the other native stem after W opens.
            g.graph.eval(); logits,_=g.forward(c[:2].cuda(),o[:2].cuda(),y[:2].cuda())
            cross={}
            for a,b in [('cfp','oct'),('oct','cfp')]:
                root=g.by_name[a+'_loss'].feature_message.current_state
                p=next(g.modules_by_name()[b+'_stage1'].parameters())
                grad=torch.autograd.grad(root,p,retain_graph=True)[0]
                cross[a+'_to_'+b]=float(grad.norm()); assert torch.isfinite(grad).all() and grad.abs().sum()>0
            report={'state':'complete','microbatch':batch,'step_seconds':step_times,'modules_changed':changed,
                    'cross_branch_gradients':cross,'gradient_norms':norm_records,'seconds':time.monotonic()-start,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'passed':True}
            write_json(out/'summary.json',report); return
        initial=evaluate(g,val,batch,seed,stop=lambda:stop_requested)
        for epoch in range(1,cfg['epochs']+1):
            g.graph.train(); opt.zero_grad(set_to_none=True); seen=0; window_count=0; window_total=min(16,len(train))
            train_ce=0.; steps=0
            for c,o,y,_ in loader(train,batch,seed,epoch-1):
                if stop_requested: raise InterruptedError('Trial stopped by controller')
                n=len(y); scale=n/window_total
                _,loss=g.forward(c.cuda(),o.cuda(),y.cuda(),loss_scale=scale)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                g.backward()
                train_ce+=float(loss.detach())/scale*n; seen+=n; window_count+=n
                if window_count==window_total:
                    clip_task_gradients(g,5.); opt.step(); opt.zero_grad(set_to_none=True)
                    steps+=1; window_count=0; window_total=min(16,len(train)-seen)
                if steps%8==0: progress(samples=seen,train_ce=train_ce/seen)
            assert seen==len(train) and window_count==0
            metrics=evaluate(g,val,batch,seed,stop=lambda:stop_requested)
            row={'epoch':epoch,'train_ce_sum':train_ce/seen,'validation':metrics}
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            progress(samples=seen,validation=metrics)
        final=evaluate(g,val,batch,seed,out/'last_predictions.npz',stop=lambda:stop_requested)
        train_final=evaluate(g,train,batch,seed,stop=lambda:stop_requested)
        if stop_requested: raise InterruptedError('Trial stopped before checkpoint')
        torch.save({'model':g.save_state(),'configuration':cfg,'epoch':epoch},out/'last.pt')
        report={'state':'complete','configuration':cfg,'initial_native_sha256':initial_hash,
                'initial':initial,'fixed_last':final,'train_last':train_final,'seconds':time.monotonic()-start,
                'parameters':info['parameters'],'trainable_parameters':info['trainable_parameters'],
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'test_used':False}
        write_json(out/'summary.json',report)
    except Exception as exc:
        oom=isinstance(exc,torch.cuda.OutOfMemoryError)
        if g is not None and opt is not None and not cfg.get('profile'):
            try:torch.save({'model':g.save_state(),'optimizer':opt.state_dict(),'epoch':epoch,'incomplete':True,'configuration':cfg},out/'interrupted.pt')
            except Exception:pass
        write_json(out/'failure.json',{'state':'oom' if oom else 'failed','error':traceback.format_exc(),
                                      'epoch':epoch,'seconds':time.monotonic()-start})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data',required=True)
    main(p.parse_args())
