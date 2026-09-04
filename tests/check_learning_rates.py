"""Check actual AdamW updates, grouping coverage and legacy rate defaults on CPU."""
import json
import torch
from radonbridge.optimization import configure_optimizer


class Graph:
    branches=('cfp','oct')
    def __init__(self):
        names=[f'{b}_stage{s}' for b in self.branches for s in range(1,5)]
        names += ['cfp_head','oct_head','bridge_0_exchange']
        self.graph=torch.nn.ModuleDict({n:torch.nn.Linear(1,1,bias=False).double() for n in names})
    def modules_by_name(self):
        return self.graph


def main():
    base={'adapt_stages':[1,2,3,4],'training_regime':'full_finetune','backbone_lr':3e-6,'weight_decay':0.}
    graph=Graph()
    cfg=base | {'head_lr':1e-5,'bridge_lr':1e-4}
    opt=configure_optimizer(graph,cfg)
    expected={name:3e-6 if '_stage' in name else 1e-4 if name.startswith('bridge_') else 1e-5 for name in graph.graph}
    assert {group['name']:group['lr'] for group in opt.param_groups} == expected
    assert all(p.requires_grad for p in graph.graph.parameters())
    before={n:m.weight.detach().clone() for n,m in graph.graph.items()}
    for p in graph.graph.parameters():p.grad=torch.ones_like(p)
    opt.step()
    for n,m in graph.graph.items():
        assert torch.allclose(before[n]-m.weight,torch.full_like(m.weight,expected[n]),rtol=1e-6,atol=1e-12)
    for group in opt.param_groups:group['lr']*=.3
    assert all(abs(g['lr']-expected[g['name']]*.3)<1e-15 for g in opt.param_groups)
    legacy=configure_optimizer(Graph(),base | {'head_bridge_lr':1e-4})
    assert all(g['lr']==(3e-6 if '_stage' in g['name'] else 1e-4) for g in legacy.param_groups)
    print(json.dumps({'split_rates':expected,'actual_adamw_updates_verified':True,'legacy_defaults_preserved':True,'schedule_ratios_preserved':True}))


if __name__=='__main__':main()
