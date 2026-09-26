"""Two-branch protocol with complete optimizer-boundary resume and plateau acceptance."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from radon_bridge.data.observed_pair import collate_observed
from radon_bridge.evaluation.paired_native import evaluate,move,replay_matches,selection_score
from radon_bridge.runtime.host_checkpoint import save,load,atomic_save,cpu_tree,save_selected,read_selected
from radon_bridge.runtime.state import atomic_write_json,file_sha256
from radon_bridge.training.convergence import Plateau

DEFAULTS=dict(backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,weight_decay=.01,
    microbatch=1,effective_batch=16,precision='fp32',clip=5.,num_workers=0,
    minimum_epochs=8,maximum_epochs=60,patience=6,min_delta=.001,lr_patience=3,lr_factor=.3)

class Schedule:
    def __init__(self,opt,cfg):
        self.opt=opt;self.cfg=cfg
        self.rule=Plateau(min_epochs=cfg['minimum_epochs'],max_epochs=cfg['maximum_epochs'],patience=cfg['patience'],
            min_delta=cfg['min_delta'],lr_patience=cfg['lr_patience'],lr_factor=cfg['lr_factor'])
    def step(self,score,epoch):
        result=self.rule.update(score,epoch)
        if result['reduce_lr']:
            for g in self.opt.param_groups:g['lr']*=self.rule.factor
        return result
    def state_dict(self):return {'config':self.cfg,'rule':vars(self.rule).copy()}
    def load_state_dict(self,state):
        if state['config']!=self.cfg:raise ValueError('Schedule changed on resume')
        self.rule.__dict__.update(state['rule'])


def validate(cfg):
    if set(cfg)!=set(DEFAULTS) or cfg['precision']!='fp32' or cfg['microbatch']<1 or cfg['effective_batch']%cfg['microbatch']:
        raise ValueError('Incomplete fixed branch training protocol')
    Schedule(None,cfg)


def train(model,train,dev,cfg,seed,out,identity,device,should_pause=lambda:False,preflight_updates=None):
    validate(cfg);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if train.split!='train' or dev.split!='development' or set(train.participant_ids)&set(dev.participant_ids):
        raise ValueError('Training/development roles overlap')
    opt=torch.optim.AdamW(model.groups(cfg['backbone_lr'],cfg['head_lr'],cfg['bridge_lr']),weight_decay=cfg['weight_decay'])
    sch=Schedule(opt,cfg);opt.zero_grad(set_to_none=True);nodes=model.node_identity()
    progress={'epoch':1,'offset':0,'updates':0,'history':[],'seconds':0.,'epoch_loss':0.,'epoch_seen':0}
    if (out/'last.pt').exists():progress=load(out/'last.pt',model=model,optimizer=opt,scheduler=sch,identity=identity,node_ids=nodes)
    start=time.monotonic()
    def status(state):atomic_write_json(dict(state=state,identity=identity,epoch=progress['epoch'],offset=progress['offset'],updates=progress['updates'],updated_at=time.time(),test_access=False),out/'status.json')
    def checkpoint():
        nonlocal start
        now=time.monotonic();progress['seconds']+=now-start;start=now
        save(out/'last.pt',model=model,optimizer=opt,scheduler=sch,identity=identity,progress=progress,node_ids=nodes)
    def loader(ds):return DataLoader(ds,batch_size=cfg['microbatch'],shuffle=False,collate_fn=collate_observed,num_workers=cfg['num_workers'],generator=torch.Generator().manual_seed(seed))
    frozen={k:v.detach().cpu().clone() for name,m in model.task.modules_by_name().items() if not name.startswith('bridge_') for k,v in ((name+':'+key,val) for key,val in m.state_dict().items())} if model.frozen else None
    def check_frozen():
        if frozen is not None:
            current={name+':'+k:v for name,m in model.task.modules_by_name().items() if not name.startswith('bridge_') for k,v in m.state_dict().items()}
            if any(not torch.equal(v.cpu(),frozen[k]) for k,v in current.items()):raise ValueError('Frozen native parameters/BN changed')
    if not (out/'best.pt').exists():
        try:
            result=evaluate(model,loader(dev),device,out/'development_predictions.npz',should_pause)
        except InterruptedError:
            # Preserve initialization and RNG before selection has completed.
            checkpoint();status('paused');return {'state':'paused'}
        sch.step(selection_score(model,result),0)
        save_selected(out/'best.pt',identity=identity,epoch=0,model=model,node_ids=nodes);checkpoint()
    launch_updates=0
    try:
        while progress['epoch']<=cfg['maximum_epochs']:
            if progress['epoch']>cfg['minimum_epochs'] and sch.rule.bad>=cfg['patience']:break
            epoch=progress['epoch'];train.set_epoch(epoch);model.train()
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            while progress['offset']<len(order):
                if should_pause():checkpoint();status('paused');return {'state':'paused'}
                block=order[progress['offset']:progress['offset']+cfg['effective_batch']]
                for batch in loader(Subset(train,block)):
                    _,loss=model(move(batch,device),len(batch['label'])/len(block))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite branch loss')
                    model.backward();progress['epoch_loss']+=float(loss.detach())*len(block);progress['epoch_seen']+=len(batch['label'])
                model.clip(cfg['clip']);opt.step();opt.zero_grad(set_to_none=True);check_frozen()
                progress['offset']+=len(block);progress['updates']+=1;launch_updates+=1;status('training')
                if progress['updates']%100==0:checkpoint()
                if should_pause() or (preflight_updates is not None and launch_updates>=preflight_updates):
                    checkpoint();status('paused');return {'state':'paused'}
            status('validating');result=evaluate(model,loader(dev),device,out/'candidate_predictions.npz',should_pause)
            decision=sch.step(selection_score(model,result),epoch)
            if decision['improved']:
                save_selected(out/'best.pt',identity=identity,epoch=epoch,model=model,node_ids=nodes)
                (out/'candidate_predictions.npz').replace(out/'development_predictions.npz')
            progress['history'].append(dict(epoch=epoch,loss=progress['epoch_loss']/progress['epoch_seen'],metrics=result,plateau=sch.rule.state()))
            progress.update(epoch=epoch+1,offset=0,epoch_loss=0.,epoch_seen=0)
            atomic_write_json(progress['history'],out/'history.json');checkpoint()
    except InterruptedError:
        checkpoint();status('paused');return {'state':'paused'}
    if sch.rule.bad<cfg['patience']:
        status('needs_review_epoch_cap');return {'state':'needs_review_epoch_cap'}
    state=read_selected(out/'best.pt',identity=identity,node_ids=nodes)
    if state['identity']!=identity or state['node_ids']!=nodes:raise ValueError('Selected identity changed')
    model.load_state_dict(state['model'],strict=True);model.eval();check_frozen()
    try:
        result=evaluate(model,loader(dev),device,out/'replay_predictions.npz',should_pause)
    except InterruptedError:
        # best is loaded for read-only acceptance. Never overwrite last with
        # these weights while its optimizer/scheduler belong to the stop state.
        status('paused');return {'state':'paused'}
    replay_matches(out/'development_predictions.npz',out/'replay_predictions.npz')
    receipt=dict(schema='radon_branch_training_v1',state='accepted',identity=identity,test_access=False,
        plateau=True,best_epoch=sch.rule.best_epoch,stop_epoch=progress['epoch']-1,metrics=result,
        frozen_native=model.frozen,frozen_native_unchanged=model.frozen,seconds=progress['seconds'],
        parameters_total=sum(p.numel() for p in model.parameters()),parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        communication_metadata={n:m.metadata for n,m in model.task.modules_by_name().items() if n.endswith('_exchange')},
        files={n:file_sha256(out/n) for n in ('best.pt','last.pt','history.json','development_predictions.npz','replay_predictions.npz')})
    atomic_write_json(receipt,out/'accepted.json');status('completed');return receipt
