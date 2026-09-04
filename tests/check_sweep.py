"""Check new sweep axes and the most parameter-heavy scheduled geometry."""
import gc
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from radonbridge.sweep import macro_f1_batch
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer, clip_task_gradients

def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(601)
    y = rng.integers(0,2,(10,40)); p = rng.integers(0,2,(10,40))
    expected = [f1_score(a,b,labels=[0,1],average='macro',zero_division=0) for a,b in zip(y,p)]
    assert np.allclose(macro_f1_batch(y,p),expected,atol=1e-12)
    protocol = json.load(open('experiments/007-combination-sweep/protocol.json'))
    assert len(protocol['recipes']) == 8 and len(protocol['bridges']) == 20
    assert len({a['id'] for a in protocol['bridges']}) == 20
    tiny = PilotGraph('radon', bridge_stages=(2,3), mixer_kernel_size=5, loss_reduction='sum')
    configure_optimizer(tiny, protocol['recipes'][0])
    assert not any(p.requires_grad for k,m in tiny.modules_by_name().items() if '_stage' in k for p in m.parameters())
    assert all(p.requires_grad for k,m in tiny.modules_by_name().items() if k.endswith('_head') for p in m.parameters())
    del tiny;gc.collect()
    gpu=[]
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
        for identifier in ['rb_mdouble_h16_s3','rb_mdouble_s23']:
            arm = next(a for a in protocol['bridges'] if a['id']==identifier)
            g = PilotGraph(arm['mode'],device='cuda',backbone='resnet18',cfp_size=224,bridge_stages=tuple(arm['stages']),
                           upsilon=tuple(arm['upsilon']),mesh_references=arm['mesh_reference'],mixer_kernel_size=arm['kernel'],loss_reduction='sum')
            opt=configure_optimizer(g,protocol['recipes'][6]);g.graph.eval();torch.cuda.reset_peak_memory_stats()
            c=torch.randn(4,2,3,224,224,device='cuda');o=torch.randn(4,2,1,32,96,96,device='cuda');y=torch.tensor([0,1,0,1],device='cuda')
            opt.zero_grad(set_to_none=True);_,loss=g.forward(c,o,y);g.backward();norms=clip_task_gradients(g);opt.step()
            assert torch.isfinite(loss) and all(torch.isfinite(v) for v in norms.values())
            gpu.append({'arm':identifier,'finite_full_batch_step':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
            del g,opt,c,o,y,loss,norms;gc.collect();torch.cuda.empty_cache()
    print(json.dumps({'vectorized_bootstrap_f1_matches_sklearn':True,'head_only_policy_verified':True,'planned_runs':38,'gpu_full_batch_checks':gpu}))

if __name__=='__main__':main()
