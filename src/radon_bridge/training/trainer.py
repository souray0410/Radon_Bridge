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
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.models.model import PilotGraph
from radon_bridge.training.convergence import Plateau
from radon_bridge.training.optimization import configure_optimizer, clip_task_gradients
from radon_bridge.training.nested import widths, set_width, training_backward, aggregate_metrics
from radon_bridge.training.frozen import validate_frozen_config, training_mode, frozen_profile


def write_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)


def bridge_configs(stages, mode='radon', *, M=16, S=64, rho=.125):
    return [{'nodes':[f'cfp_stage{s}',f'oct_stage{s}'],'M':M,'S':S,'rho':rho,'mode':mode} for s in stages]


def loader(data, batch, seed, epoch=None):
    return DataLoader(data,batch_size=batch,shuffle=epoch is not None,num_workers=2,pin_memory=True,
                      generator=torch.Generator().manual_seed(seed+(epoch or 0)))


@torch.no_grad()
def evaluate_one_width(g, data, batch, seed, output=None, stop=None):
    g.graph.eval(); scores={k:[] for k in g.output_names}; labels=[]; ids=[]; ratios={}
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
    return {'tasks':tasks,'mean_task_macro_f1':float(np.mean([tasks[k]['macro_f1'] for k in g.branches])),
            'delta_over_feature_l2':{k:math_sqrt(a/b) if b else 0. for k,(a,b) in ratios.items()}}


def evaluate(g,data,batch,seed,output=None,stop=None):
    rhos=widths(g)
    if rhos is None:return evaluate_one_width(g,data,batch,seed,output,stop)
    values={}
    try:
        for rho in rhos:
            set_width(g,rho)
            name='rho1_'+str(round(1/rho))
            path=Path(output).with_name(Path(output).stem+'_'+name+'.npz') if output else None
            values[name]=evaluate_one_width(g,data,batch,seed,path,stop)
            if output and rho==max(rhos):
                import shutil
                shutil.copyfile(path,output)
    finally:set_width(g,max(rhos))
    return aggregate_metrics(values)


def selection_score(metrics,cfg):
    return metrics['tasks']['fusion']['macro_f1'] if cfg.get('selection_metric') == 'fusion_macro_f1' else metrics['mean_task_macro_f1']


def math_sqrt(x): return float(np.sqrt(x))


def parameter_hash(g):
    h=hashlib.sha256()
    for name,module in sorted(g.modules_by_name().items()):
        if name.startswith(('bridge_', 'fusion_')):continue
        h.update(name.encode())
        for key,p in sorted(module.state_dict().items()):
            h.update(key.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(3); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from radon_bridge.runtime.artifacts import relocate
    cfg=relocate(json.loads(Path(args.config).read_text())); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
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
    frozen=validate_frozen_config(cfg)
    recipe={'adapt_stages':[1,2,3,4],'backbone_lr':cfg['backbone_lr'],
            'head_lr':cfg.get('head_lr',1e-4),'bridge_lr':cfg.get('bridge_lr',1e-4),
            'weight_decay':.01,'training_regime':'bridge_only' if frozen else 'full_finetune'}
    g=None; opt=None; epoch=0; start=time.monotonic()
    try:
        g=PilotGraph(bridge_configs=cfg['bridges'],seed=seed,device='cuda',task_fusion=cfg.get('task_fusion'))
        if g.task_fusion is not None and (cfg.get('selection_metric') != 'fusion_macro_f1' or cfg.get('training_stage') not in ('communication','host_augmentation')):
            raise ValueError('Task fusion requires explicit fusion selection and communication stage')
        basis_artifacts=[]
        for group in g.communication_groups:
            module=g.modules_by_name()[group['exchange_edge_name']]
            basis_artifacts.extend(module.export_fixed_bases(out.parent/'bases'))
        if basis_artifacts:write_json(out/'basis_manifest.json',basis_artifacts)
        parents={}
        if cfg.get('training_stage')=='communication':
            if set(cfg.get('parent_checkpoints',{})) != {'cfp','oct'}:
                raise ValueError('Communication training requires both independently trained modality checkpoints')
            for branch, parent in cfg['parent_checkpoints'].items():
                checkpoint=Path(parent['path'])
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=parent['sha256']:
                    raise ValueError('Parent checkpoint hash mismatch')
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if saved.get('branch')!=branch or saved.get('training_stage')!='independent' or saved.get('seed')!=seed or saved.get('stop_reason')!='validation_plateau':
                    raise ValueError('Not a matching independently trained modality checkpoint')
                g.load_native_state(saved['model'],branch=branch)
                parents[branch]=parent
        elif cfg.get('training_stage')=='host_augmentation':
            from radon_bridge.runtime.artifacts import sha256
            parent=cfg['host_checkpoint']; checkpoint=Path(parent['path'])
            if sha256(checkpoint)!=parent['sha256']: raise ValueError('Host checkpoint SHA mismatch')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            original=saved['configuration']
            for key in ('seed','backbone_lr','task_fusion','selection_metric'):
                if original.get(key)!=cfg.get(key): raise ValueError('Host protocol mismatch: '+key)
            if original['bridges']!=cfg['bridges'][:1]: raise ValueError('Complete host adapter must be preserved')
            g.load_complete_state(saved['model'],allow_new_bridge=len(cfg['bridges'])==2)
        elif cfg.get('training_stage')=='independent' and cfg['bridges']:
            raise ValueError('Independent pretraining cannot contain a bridge')
        opt=configure_optimizer(g,recipe)
        initial_hash=parameter_hash(g)
        initial_learning_rates={group['name']:group['lr'] for group in opt.param_groups}
        info={'configuration':cfg,'initial_native_sha256':initial_hash,'groups':g.communication_groups,
              'initial_learning_rates':initial_learning_rates,
              'parameters':sum(p.numel() for p in g.graph.parameters()),
              'trainable_parameters':sum(p.numel() for p in g.graph.parameters() if p.requires_grad),
              'communication_parameters':sum(p.numel() for n,m in g.modules_by_name().items() if n.startswith('bridge_') for p in m.parameters()),
              'parent_checkpoints':parents,'batchnorm_policy':'train','initialization':'CFP ImageNet; OCT inflated ImageNet, not OCT-specific pretraining'}
        if g.task_fusion is not None:
            info.update(selection_policy='fusion macro-F1',loss_policy='CE_CFP + CE_OCT + CE_fusion',task_fusion=cfg['task_fusion'],fusion_parameters=sum(p.numel() for p in g.modules_by_name()['fusion_head'].parameters()))
        if widths(g):
            info.update(nested_widths=list(widths(g)),batchnorm_policy='per-width training batch statistics; arithmetic mean of running updates, one counter increment per participant batch',selection_policy='one checkpoint maximizing equal mean of six branch-width F1 values',optimizer_updates_per_batch=1)
        if frozen:
            info.update(training_regime='bridge_only',batchnorm_policy='native eval; parameters and buffers fixed',
                        backbone_lr_applicable=False,head_lr_applicable=False)
        elif any(not p.requires_grad for p in g.graph.parameters()): raise AssertionError('Unexpected frozen parameter')
        write_json(out/'model.json',info)
        from radon_bridge.training.optimization import apply_clipping
        apply_clipping(g,cfg)
        batch=cfg['microbatch']; torch.cuda.reset_peak_memory_stats()
        def progress(**kw):
            write_json(out/'progress.json',{'pid':os.getpid(),'epoch':epoch,'elapsed_seconds':time.monotonic()-start,**kw})
        if cfg.get('profile'):
            if frozen:
                report=frozen_profile(g,opt,train,batch,seed,progress)
                report.update(seconds=time.monotonic()-start,configuration=cfg)
                write_json(out/'summary.json',report);return
            g.graph.train(); c,o,y,profile_keys=next(iter(loader(train,batch,seed,0)))
            before={k:[p.detach().clone() for p in m.parameters()] for k,m in g.modules_by_name().items() if list(m.parameters())}
            norm_records=[]; step_times=[]
            for step in range(3):
                if stop_requested: raise InterruptedError('Profile stopped')
                opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); tick=time.monotonic()
                loss=training_backward(g,c.cuda(),o.cuda(),y.cuda())
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
                grad=torch.autograd.grad(root,p,retain_graph=True,allow_unused=True)[0]
                if grad is None: grad=torch.zeros_like(p)
                cross[a+'_to_'+b]=float(grad.norm()); assert torch.isfinite(grad).all()
                allowed=any(x.get('family','radon')!='radon' or (x['mode']!='self' and
                    ('cross_edges' not in x or [b+'_stage3',a+'_stage3'] in x['cross_edges'])) for x in cfg['bridges'])
                if not allowed:assert grad.abs().sum()==0
                else:assert grad.abs().sum()>0
            if g.task_fusion is not None:
                fusion_root=g.by_name['fusion_loss'].feature_message.current_state
                for branch in g.branches:
                    p=next(g.modules_by_name()[branch+'_stage1'].parameters())
                    grad=torch.autograd.grad(fusion_root,p,retain_graph=True)[0]
                    assert torch.isfinite(grad).all() and grad.norm()>0
                    cross['fusion_to_'+branch]=float(grad.norm())
            report={'state':'complete','microbatch':batch,'step_seconds':step_times,'modules_changed':changed,
                    'cross_branch_gradients':cross,'gradient_norms':norm_records,'seconds':time.monotonic()-start,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'passed':True,'nested_rhos':list(widths(g)) if widths(g) else None}
            if cfg.get('save_profile_checkpoint'):
                torch.save({'model':g.save_state(),'configuration':cfg},out/'profile_selected.pt')
                if widths(g):
                    from radon_bridge.evaluation.export import export_widths
                    exports=export_widths(g.save_state(),cfg,out)
                    g.graph.eval()
                    with torch.no_grad():
                        for view in exports:
                            set_width(g,view['rho']);logits,_=g.forward(c.cuda(),o.cuda(),y.cuda())
                            np.savez(Path(view['directory'])/'profile_predictions.npz',ids=np.asarray(profile_keys),y=y.numpy(),**{k:v.softmax(1).cpu().numpy() for k,v in logits.items()})
                    set_width(g,max(widths(g)))
            if g.task_fusion is not None or cfg.get('measure_latency'):
                import subprocess
                g.graph.eval()
                before_state=g.save_state();rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
                context=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
                gpu_uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
                timings=[];latency_batch=(c.cuda(),o.cuda(),y.cuda())
                with torch.no_grad():
                    for index in range(60):
                        torch.cuda.synchronize();tick=time.monotonic();g.forward(*latency_batch);torch.cuda.synchronize()
                        if index>=10:timings.append(1000*(time.monotonic()-tick))
                assert all(torch.equal(v,g.modules_by_name()[n].state_dict()[k].cpu()) for n,d in before_state.items() for k,v in d.items())
                assert torch.equal(rng,torch.random.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
                visible=int(os.environ['CUDA_VISIBLE_DEVICES'])
                uuid=next(line.split(',')[1].strip() for line in gpu_uuids.splitlines() if int(line.split(',')[0])==visible)
                other=[line for line in context.splitlines() if uuid in line and int(line.split(',')[0])!=os.getpid()]
                report['latency']=dict(phase='preflight after three optimization steps',batch=16,warmup=10,repeats=50,median_ms=float(np.median(timings)),q25_ms=float(np.quantile(timings,.25)),q75_ms=float(np.quantile(timings,.75)),timings_ms=timings,other_processes_on_device=other,interfered=bool(other),parameters_BN_RNG_preserved=True)
            write_json(out/'summary.json',report); return
        initial=evaluate(g,val,batch,seed,out/'initial_predictions.npz',stop=lambda:stop_requested)
        if cfg.get('training_stage')=='host_augmentation':
            with np.load(Path(cfg['host_checkpoint']['path']).parent/'selected_predictions.npz',allow_pickle=False) as previous, np.load(out/'initial_predictions.npz',allow_pickle=False) as current:
                assert np.array_equal(previous['ids'],current['ids']) and np.array_equal(previous['y'],current['y'])
                assert all(np.allclose(previous[k],current[k],rtol=1e-5,atol=1e-6) for k in g.output_names)
            write_json(out/'checkpoint_acceptance.json',{'strict_host_load':True,'initial_predictions_match':True,'optimizer_reset_for_all_arms':True})
        if parents:
            for branch,parent in parents.items():
                with np.load(Path(parent['path']).parent/'selected_predictions.npz',allow_pickle=False) as previous, np.load(out/'initial_predictions.npz',allow_pickle=False) as current:
                    assert np.array_equal(previous['ids'],current['ids']) and np.array_equal(previous['y'],current['y'])
                    if not np.allclose(previous[branch],current[branch],rtol=1e-5,atol=1e-6):
                        raise AssertionError('Stage two initial prediction differs from stage one checkpoint')
            write_json(out/'checkpoint_acceptance.json',{'strict_load':True,'parent_hashes_verified':True,'initial_predictions_match':True,'optimizer_reset_for_all_arms':True})
        policy=cfg['convergence']
        independent=cfg['training_stage']=='independent'
        monitors={k:Plateau(**policy) for k in (g.branches if independent else ('joint',))}
        # Stage two may retain the loaded checkpoint if every update degrades validation.
        selected_states={}
        for key,monitor in monitors.items():
            if not independent:
                monitor.update(selection_score(initial,cfg),0)
                selected_states[key]=g.save_state()
        converged=False; epoch_times=[]
        for epoch in range(1,policy['max_epochs']+1):
            epoch_start=time.monotonic()
            training_mode(g,frozen); opt.zero_grad(set_to_none=True); seen=0; window_count=0; window_total=min(16,len(train))
            train_ce=0.; steps=0
            for c,o,y,_ in loader(train,batch,seed,epoch-1):
                if stop_requested: raise InterruptedError('Trial stopped by controller')
                n=len(y); scale=n/window_total
                loss=training_backward(g,c.cuda(),o.cuda(),y.cuda(),scale)
                train_ce+=float(loss.detach())/scale*n; seen+=n; window_count+=n
                if window_count==window_total:
                    clip_task_gradients(g,5.); opt.step(); opt.zero_grad(set_to_none=True)
                    steps+=1; window_count=0; window_total=min(16,len(train)-seen)
                if steps%8==0: progress(samples=seen,train_ce=train_ce/seen)
            assert seen==len(train) and window_count==0
            if frozen:assert parameter_hash(g)==initial_hash,'Frozen native parameters or BN buffers changed'
            metrics=evaluate(g,val,batch,seed,stop=lambda:stop_requested)
            flags={}
            for key,monitor in monitors.items():
                score=metrics['tasks'][key]['macro_f1'] if independent else selection_score(metrics,cfg)
                flags[key]=monitor.update(score,epoch)
                if flags[key]['improved']:
                    selected_states[key]={name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
                                          for name,module in g.modules_by_name().items()
                                          if not independent or name.startswith(key+'_')}
                if flags[key]['reduce_lr']:
                    for group in opt.param_groups:
                        if not independent or group['name'].startswith(key+'_'):group['lr']*=monitor.factor
            epoch_times.append(time.monotonic()-epoch_start)
            row={'epoch':epoch,'train_ce_sum':train_ce/seen,'validation':metrics,
                 'plateau':{k:m.state() for k,m in monitors.items()},
                 'learning_rates':{group['name']:group['lr'] for group in opt.param_groups}}
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            progress(samples=seen,validation=metrics,plateau=row['plateau'])
            if all(f['plateau'] for f in flags.values()):
                converged=True;break
        final=evaluate(g,val,batch,seed,out/'last_predictions.npz',stop=lambda:stop_requested)
        train_final=evaluate(g,train,batch,seed,stop=lambda:stop_requested)
        if stop_requested: raise InterruptedError('Trial stopped before checkpoint')
        # Preserve stopping-point state separately from selected development checkpoint.
        torch.save({'model':g.save_state(),'optimizer':opt.state_dict(),'configuration':cfg,'epoch':epoch,
                    'stop_reason':'validation_plateau' if converged else 'epoch_cap'},out/'last.pt')
        if independent:
            for branch in g.branches:g.load_native_state(selected_states[branch],branch)
        else:
            for name,module in g.modules_by_name().items():module.load_state_dict(selected_states['joint'][name],strict=True)
        selected=evaluate(g,val,batch,seed,out/'selected_predictions.npz',stop=lambda:stop_requested)
        state=g.save_state()
        if frozen:assert parameter_hash(g)==initial_hash,'Selected frozen native state changed'
        torch.save({'model':state,'configuration':cfg,'selection':{k:m.state() for k,m in monitors.items()}},out/'selected.pt')
        modality_checkpoints={}
        if widths(g):
            from radon_bridge.evaluation.export import export_widths
            export_widths(state,cfg,out,{k:m.state() for k,m in monitors.items()})
        if independent and converged:
            for branch in g.branches:
                path=out/(branch+'.pt')
                torch.save({'model':{k:v for k,v in state.items() if k.startswith(branch+'_')},
                            'branch':branch,'training_stage':'independent','seed':seed,
                            'epoch':monitors[branch].best_epoch,'stop_reason':'validation_plateau',
                            'source_commit':cfg.get('source_commit')},path)
                modality_checkpoints[branch]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        report={'state':'complete' if converged else 'incomplete','stop_reason':'validation_plateau' if converged else 'epoch_cap',
                'converged_by_policy':converged,'epochs_ran':epoch,'epoch_seconds':epoch_times,
                'selection':{k:m.state() for k,m in monitors.items()},'configuration':cfg,'initial_native_sha256':initial_hash,
                'initial_learning_rates':initial_learning_rates,
                'parent_checkpoints':parents,'modality_checkpoints':modality_checkpoints,'initial':initial,
                'stopping_metrics':final,'selected':selected,'train_stopping_metrics':train_final,'seconds':time.monotonic()-start,
                'parameters':info['parameters'],'trainable_parameters':info['trainable_parameters'],
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'test_used':False}
        if frozen:report.update(training_regime='bridge_only',native_parameters_and_buffers_unchanged=True,
                                selected_native_sha256=parameter_hash(g),optimizer_parameter_groups=list(initial_learning_rates))
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
