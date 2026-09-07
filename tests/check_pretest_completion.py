"""Matrix reuse, contrast algebra, and frozen self/QR optimizer invariants."""
import copy,json,tempfile
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
from radonbridge.pretest_study import matrices,contrast_definitions
from radonbridge.geometry_study import configuration
from radonbridge.frozen_training import validate_frozen_config,bridge_only_optimizer,training_mode
from radonbridge.bridge import LinearMixer,BridgeExchange,FeatureSpec
from radonbridge.svd_basis import save_basis,save_random_basis

torch.set_num_threads(3)
with tempfile.TemporaryDirectory() as tmp:
    refs={key:save_basis(torch.eye(8,dtype=torch.float64),1,tmp,key,3416,{'synthetic':True}) for key in ('a','b')}
    qrrefs={key:save_random_basis(ref,Path(tmp)/'qr') for key,ref in refs.items()}
    specs=[FeatureSpec('a',8,(3,4)),FeatureSpec('b',8,(2,3,4))]
    xs=[torch.randn(2,8,*s.shape,requires_grad=True) for s in specs]
    for comp,files in [('fixed_svd_channel',refs),('fixed_random_orthogonal_channel',qrrefs)]:
        for mode in ('radon','linear_resample','self'):
            for rank in (1,2,4):
                kw=dict(M=4,S=7,rho=rank/8,compression=comp,basis_files=files,mode=mode)
                old=BridgeExchange(specs,**kw).float()
                new=BridgeExchange(specs,r=rank,h=rank*4,**kw).float()
                with torch.no_grad():old.mixer.conv.weight.normal_(std=.01);old.mixer.conv.weight.mul_(old.mixer.mask)
                new.load_state_dict(old.state_dict(),strict=True)
                assert torch.equal(old(*xs),new(*xs))
                ga=torch.autograd.grad(old(*xs).square().sum(),xs)
                gb=torch.autograd.grad(new(*xs).square().sum(),xs)
                assert all(torch.equal(a,b) for a,b in zip(ga,gb))
seeds=(3416,3417,3418)
parents={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp','oct')} for s in seeds}
bases={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp_stage3','oct_stage3')} for s in seeds}
qr=copy.deepcopy(bases)
for refs in qr.values():
    for r in refs.values():r['sha256']+='qr'
geo,frozen=matrices(parents,bases,qr)
assert len(geo)==24 and len(frozen)==54
assert all(validate_frozen_config(r['configuration']) for r in frozen)
for field,val in [('r',8),('h',9),('rho',.5),('compression','learned_channel')]:
    c=copy.deepcopy(frozen[0]['configuration']);c['bridges'][0][field]=val
    try:validate_frozen_config(c)
    except AssertionError:pass
    else:raise AssertionError(field)
old=[]
for M,r in ((16,32),(32,16),(32,32),(64,16)):
    for mode in ('radon','linear_resample'):
        for seed in seeds:
            for lr in (3e-5,6e-5):
                old.append(dict(id=f'{M}_{r}_{mode}_{seed}_{lr}',structure=dict(M=M,r=r,S=64,k=3,mode=mode)))
full=[dict(id=f'full_{b}_{m}_{s}_{r}',arm=b+'_'+m) for b in ('svd','qr') for m in ('radon','resample') for s in seeds for r in (16,32,64)]
defs=contrast_definitions(geo+old,frozen,full)
assert len(defs)==17 and all(abs(sum(d['weights'].values()))<1e-10 for d in defs)
assert len({d['id'] for d in defs})==17
for compression in ('fixed_svd_channel','fixed_random_orthogonal_channel'):
    for self_only in (False,True):
        ex=nn.Module();ex.compression=compression;ex.mixer=LinearMixer([2,2],3,self_only=self_only)
        ms=nn.ModuleDict({'cfp_head':nn.Linear(2,2),'oct_head':nn.Linear(2,2),'bridge_0_exchange':ex})
        g=SimpleNamespace(task_fusion=None,communication_groups=[{}],modules_by_name=lambda:dict(ms.items()),graph=ms)
        opt=bridge_only_optimizer(g,dict(bridge_lr=1e-4));training_mode(g,True)
        x=torch.randn(2,2,5,requires_grad=True);y=torch.randn(2,2,5,requires_grad=True)
        for _ in range(3):
            opt.zero_grad();a,b=ex.mixer(x,y);(a.square().sum()+b.square().sum()+a.sum()+b.sum()).backward();opt.step()
        a,b=ex.mixer(x,y);grad=torch.autograd.grad(a.sum(),y,allow_unused=True)[0]
        norm=0. if grad is None else float(grad.norm())
        assert norm==0 if self_only else norm>0
        assert all(not p.requires_grad for n,m in ms.items() if not n.startswith('bridge_') for p in m.parameters())
        assert torch.count_nonzero(ex.mixer.conv.weight.detach()*(1-ex.mixer.mask))==0
print(json.dumps(dict(passed=True,geometry=24,frozen=54,reused_frozen=6,new_training=72,contrasts=17)))
