"""Actual CFP/OCT MHD graph: frozen native stages still pass gradients to the bridge."""
import json,gc
import torch
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer,clip_task_gradients
from radonbridge.frozen_training import training_mode
from radonbridge.experiment import parameter_hash
from radonbridge.nested_training import training_backward
from radonbridge.diagnostics import release_forward_graph
from scripts.run_dependency_supplement import frozen_rows
from scripts.geometry_evidence import dependencies

torch.set_num_threads(3);torch.manual_seed(3416)
parents,bases=dependencies();cfg=frozen_rows(parents,bases)[0]['configuration']
g=PilotGraph(bridge_configs=cfg['bridges'],seed=3416,device='cpu')
for branch,ref in parents[3416].items():
    saved=torch.load(ref['path'],map_location='cpu',weights_only=False);g.load_native_state(saved['model'],branch)
opt=configure_optimizer(g,dict(training_regime='bridge_only',bridge_lr=1e-4,weight_decay=.01))
before=parameter_hash(g);native_ids={k:g.by_name[k].id for k in ('cfp_stage3','oct_stage3')}
c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
training_mode(g,True)
for step in range(3):
    opt.zero_grad(set_to_none=True);training_backward(g,c,o,y);clip_task_gradients(g);opt.step()
    assert parameter_hash(g)==before
    assert native_ids=={k:g.by_name[k].id for k in native_ids}
g.graph.eval();weight=g.modules_by_name()['bridge_0_exchange'].mixer.conv.weight
assert weight.abs().sum()>0
opt.zero_grad(set_to_none=True);_,loss=g.forward(c,o,y)
reference=torch.autograd.grad(loss,weight,retain_graph=True)[0].clone();g.backward()
assert torch.allclose(weight.grad,reference,atol=1e-7,rtol=1e-5)
state=g.save_state();release_forward_graph(g);g.load_complete_state(state)
logits,_=g.forward(c,o,y);expected={k:v.detach().clone() for k,v in logits.items()};release_forward_graph(g)
g.load_complete_state(state);logits,_=g.forward(c,o,y)
assert all(torch.equal(expected[k],v) for k,v in logits.items()) and parameter_hash(g)==before
print(json.dumps(dict(passed=True,full_CFP_OCT_MHD=True,updates=3,native_parameters_BN_unchanged=True,
                     bridge_gradient_matches_autograd=True,strict_checkpoint_reload=True,original_node_ids=True)))
