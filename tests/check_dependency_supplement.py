"""Frozen optimizer/BN, residual switches, scope and default-policy regression."""
import json,copy
from types import SimpleNamespace
import torch
from torch import nn
from radonbridge.bridge import LinearMixer
from radonbridge.optimization import configure_optimizer,clip_task_gradients
from radonbridge.frozen_training import validate_frozen_config,training_mode
from radonbridge.component_ablation import component_switch
from radonbridge.diagnostics import read_only
from radonbridge.experiment import parameter_hash
from scripts.run_dependency_supplement import frozen_rows

parents={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp','oct')} for s in (3416,3417,3418)}
bases={s:{k:dict(path='/old/'+k,sha256=str(s)+k) for k in ('cfp_stage3','oct_stage3')} for s in parents}
rows=frozen_rows(parents,bases)
assert len(rows)==6 and all(validate_frozen_config(r['configuration']) for r in rows)
assert not validate_frozen_config({})
for key,value in [('head_lr',1e-4),('backbone_lr',3e-5),('optimization_mode','unknown')]:
    bad=dict(rows[0]['configuration']);bad[key]=value
    try:validate_frozen_config(bad)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Invalid frozen regime accepted')

class Exchange(nn.Module):
    def __init__(self):
        super().__init__();self.compression='fixed_svd_channel';self.mixer=LinearMixer([2,2],3)
ex=Exchange();modules=nn.ModuleDict({'cfp_stage1':nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4)),
                                   'oct_stage1':nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4)),
                                   'cfp_head':nn.Linear(4,2),'oct_head':nn.Linear(4,2),'bridge_0_exchange':ex})
g=SimpleNamespace(graph=modules,modules_by_name=lambda:dict(modules.items()),branches=('cfp','oct'),head_mode='separate',task_fusion=None,communication_groups=[{}])
recipe=dict(training_regime='bridge_only',bridge_lr=1e-4,weight_decay=.01)
opt=configure_optimizer(g,recipe);training_mode(g,True);before=parameter_hash(g)
for step in range(3):
    opt.zero_grad(set_to_none=True)
    x=modules['cfp_stage1'](torch.randn(16,4));y=modules['oct_stage1'](torch.randn(16,4))
    delta=ex.mixer(x[:,:2,None].expand(-1,-1,3),y[:,:2,None].expand(-1,-1,3))[0].mean(-1)
    loss=delta.square().mean()+delta.mean();loss.backward();clip_task_gradients(g,5.);opt.step()
    assert parameter_hash(g)==before
assert ex.mixer.conv.weight.abs().sum()>0
assert all(p.grad is None for name,m in modules.items() if not name.startswith('bridge_') for p in m.parameters())
full=dict(training_regime='full_finetune',adapt_stages=[1,2,3,4],backbone_lr=3e-5,head_lr=1e-4,bridge_lr=1e-4)
configure_optimizer(g,full);training_mode(g,False)
assert all(p.requires_grad for p in modules.parameters()) and modules['cfp_stage1'][1].training

class Packet(nn.Module):
    def __init__(self,gain,delta_only=False):
        super().__init__();self.gain=nn.Parameter(torch.tensor(gain));self.delta_only=delta_only
    def forward(self,*xs):
        self.latest_inputs=xs;self.latest_deltas=tuple(x*self.gain for x in xs)
        return torch.cat([(dx if self.delta_only else x+dx).flatten(1) for x,dx in zip(xs,self.latest_deltas)],1)
host=Packet(2.);new=Packet(-.5,True);loss=nn.Identity();loss.scale=1.
ms=nn.ModuleDict({'bridge_0_exchange':host,'bridge_1_exchange':new,'bridge_parallel_merge':nn.Identity(),'task_loss_sum':loss})
h=SimpleNamespace(graph=ms,modules_by_name=lambda:dict(ms.items()))
xs=(torch.randn(2,4,3),torch.randn(2,4,2,2));flat=torch.cat([x.flatten(1) for x in xs],1)
with read_only(h):
    for host_on,new_on in ((True,True),(True,False),(False,True),(False,False)):
        with component_switch(h,host_on,new_on):
            got=host(*xs)+new(*xs)
            assert torch.allclose(got,flat*(1+2*host_on-.5*new_on),atol=1e-6)
    try:
        with component_switch(h,False,False):raise RuntimeError('injected')
    except RuntimeError:pass
    assert len(host._forward_hooks)==len(new._forward_hooks)==0
    assert torch.allclose(host(*xs)+new(*xs),flat*2.5,atol=1e-6)
print(json.dumps(dict(passed=True,frozen_trials=6,component_states=4,checks=['native_parameters_and_BN_fixed','mixer_updates',
 'frozen_optimizer_unique','default_full_training_restored','scope_guard','residual_switch_equations','exception_hook_cleanup','read_only_state_preservation'])))
