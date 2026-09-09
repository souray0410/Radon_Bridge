"""Explicit supplemental regime; default full fine-tuning is unchanged."""
import torch


def validate_frozen_config(cfg):
    mode=cfg.get('optimization_mode','full_finetune')
    if mode not in ('full_finetune','bridge_only'):raise ValueError('Unknown optimization_mode')
    if mode=='full_finetune':return False
    assert cfg['training_stage']=='communication' and not cfg.get('task_fusion')
    assert cfg['backbone_lr'] is None and cfg.get('head_lr') is None
    assert cfg['bridge_lr']==1e-4 and cfg['microbatch']==cfg['effective_batch']==16
    assert len(cfg['bridges'])==1
    b=cfg['bridges'][0]
    assert b.get('compression') in ('fixed_svd_channel','fixed_random_orthogonal_channel')
    assert b['mode'] in ('radon','linear_resample','self')
    assert (b['M'],b['S'],b['kernel_size'])==(32,64,3)
    assert b['r'] in (16,32,64) and b['h']==32*b['r'] and b['rho']==b['r']/256
    return True


def bridge_only_optimizer(g,recipe):
    assert g.task_fusion is None and len(g.communication_groups)==1
    ex=g.modules_by_name()['bridge_0_exchange']
    assert ex.compression in ('fixed_svd_channel','fixed_random_orthogonal_channel')
    groups=[]
    for name,module in g.modules_by_name().items():
        enabled=name.startswith('bridge_')
        ps=list(module.parameters())
        for p in ps:p.requires_grad_(enabled)
        if enabled and ps:groups.append(dict(params=ps,name=name,lr=recipe['bridge_lr']))
    ids=[id(p) for group in groups for p in group['params']]
    assert len(ids)==len(set(ids))==1,'Fixed bridge must learn only its bias-free mixer weight'
    assert set(ids)=={id(p) for p in ex.mixer.parameters()}
    return torch.optim.AdamW(groups,weight_decay=recipe.get('weight_decay',.01))


def training_mode(g,frozen):
    g.graph.train()
    if frozen:
        for name,module in g.modules_by_name().items():
            if not name.startswith('bridge_'):module.eval()


def frozen_profile(g,opt,train,batch,seed,progress):
    from radon_bridge.training.trainer import loader, parameter_hash
    from radon_bridge.training.nested import training_backward
    from radon_bridge.training.optimization import clip_task_gradients
    before=parameter_hash(g);modules=g.modules_by_name();ex=modules['bridge_0_exchange']
    buffers={k:v.detach().clone() for k,v in ex.named_buffers()}
    weights=ex.mixer.conv.weight.detach().clone()
    c,o,y,_=next(iter(loader(train,batch,seed,0)));c=c.cuda();o=o.cuda();y=y.cuda()
    training_mode(g,True)
    for step in range(3):
        opt.zero_grad(set_to_none=True);loss=training_backward(g,c,o,y)
        norms=clip_task_gradients(g,5.);assert list(norms)==['communication']
        assert ex.mixer.conv.weight.grad is not None and torch.isfinite(ex.mixer.conv.weight.grad).all()
        opt.step();progress(step=step+1,train_loss=float(loss))
        assert parameter_hash(g)==before
        assert all(p.grad is None for name,m in modules.items() if not name.startswith('bridge_') for p in m.parameters())
    assert not torch.equal(weights,ex.mixer.conv.weight)
    assert all(torch.equal(buffers[k],v) for k,v in ex.named_buffers())
    g.graph.eval();c=c[:2].detach().requires_grad_();o=o[:2].detach().requires_grad_()
    g.forward(c,o,y[:2]);cross={}
    for branch,source in [('cfp',o),('oct',c)]:
        loss=g.by_name[branch+'_loss'].feature_message.current_state
        grad=torch.autograd.grad(loss,source,retain_graph=True,allow_unused=True)[0]
        norm=0. if grad is None else float(grad.norm())
        assert grad is None or torch.isfinite(grad).all()
        self_only=all(group['mode']=='self' for group in g.communication_groups)
        assert norm==0. if self_only else norm>0.
        cross[branch]=norm
    if self_only:
        assert torch.count_nonzero(ex.mixer.conv.weight.detach()*(1-ex.mixer.mask))==0
    assert parameter_hash(g)==before
    return dict(state='complete',passed=True,microbatch=batch,optimizer_updates=3,
                native_parameters_and_buffers_unchanged=True,fixed_bridge_buffers_unchanged=True,
                only_mixer_updated=True,cross_source_input_gradient_norms=cross,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,test_used=False)
