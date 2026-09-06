"""Exact legacy graph compatibility and importing a differently named MHD network."""
import json,subprocess,types
import torch
from torch import nn
from radonbridge.graph import MHDBuilder
from radonbridge.model import PilotGraph
from radonbridge.network_interface import NativeNetwork
from radonbridge.mhd_task import MHDTaskGraph

class Pool(nn.Module):
    def forward(self,x):return x.flatten(2).mean(-1)
class AddLoss(nn.Module):
    def forward(self,a,b):return a+b

def other_network():
    b=MHDBuilder();node,edge=b.node,b.edge
    inputs={k:node(k) for k in ('signal','volume','labels')};source_names={};outputs={}
    for source,channels,conv in [('signal',4,nn.Conv1d(1,4,3,padding=1)),('volume',6,nn.Conv3d(1,6,3,padding=1))]:
        f=node(source+'_embedding');edge(source+'_encoder',conv,[inputs[source]],[f])
        p=node(source+'_pooled');edge(source+'_pool',Pool(),[f],[p])
        o=node(source+'_prediction');edge(source+'_head',nn.Linear(channels,2),[p],[o]);outputs[source]=source+'_prediction'
        loss=node(source+'_objective');edge(source+'_criterion',nn.CrossEntropyLoss(),[o,inputs['labels']],[loss])
        source_names[source]=tuple(e.name for e in b.edges if e.name.startswith(source+'_'))
    total=node('combined_objective');edge('combine',AddLoss(),[b.by_name[s+'_objective'].id for s in ('signal','volume')],[total])
    return NativeNetwork(b,('signal','volume','labels'),outputs,'combined_objective',
                         {'signal':torch.zeros(2,1,13),'volume':torch.zeros(2,1,3,4,5),'labels':torch.zeros(2,dtype=torch.long)},
                         ('signal','volume'),tuple(e.name for e in b.edges),source_names)

def check():
    torch.set_num_threads(3)
    old=types.ModuleType('radonbridge._legacy_model_interface');old.__package__='radonbridge'
    exec(compile(subprocess.check_output(['git','show','18c804a:radonbridge/model.py'],text=True),'legacy_model','exec'),old.__dict__)
    cases=[({},[]),({},[dict(nodes=['cfp_stage3','oct_stage3'],M=4,S=11,rho=.125,mode='radon')]),
           ({'task_fusion':dict(pooling='mean',hidden_dimension=256,attention_dimension=128)},[dict(nodes=['cfp_stage3','oct_stage3'],family='mmtm',reduction_ratio=8)])]
    for extra,bridges in cases:
        a=old.PilotGraph(seed=3416,bridge_configs=bridges,**extra);b=PilotGraph(seed=3416,bridge_configs=bridges,**extra)
        sa=a.save_state();sb=b.save_state();assert sa.keys()==sb.keys()
        assert all(sa[n].keys()==sb[n].keys() and all(torch.equal(t,sb[n][k]) for k,t in sa[n].items()) for n in sa)
        assert a.steps==b.steps and a.forward_levels==b.forward_levels and a.backward_levels==b.backward_levels
        assert {k:v.id for k,v in a.by_name.items()}=={k:v.id for k,v in b.by_name.items()}
        a.graph.eval();b.graph.eval();c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
        pa,la=a.forward(c,o,y);pb,lb=b.forward(c,o,y);assert torch.equal(la,lb) and all(torch.equal(pa[k],pb[k]) for k in pa)
        a.backward();b.backward()
        assert all(torch.equal(x.grad,z.grad) for x,z in zip(a.graph.parameters(),b.graph.parameters()))
        del a,b,sa,sb,pa,pb,la,lb
    native=other_network();ids={k:n.id for k,n in native.builder.by_name.items()}
    graph=MHDTaskGraph(native,bridge_configs=[dict(nodes=['signal_embedding','volume_embedding'],M={'signal_embedding':1,'volume_embedding':6},S=9,rho=.5,mode='radon',kernel_size=5)])
    with torch.no_grad():graph.modules_by_name()['bridge_0_exchange'].mixer.conv.weight.normal_(std=.01)
    values={'signal':torch.randn(2,1,13),'volume':torch.randn(2,1,3,4,5),'labels':torch.tensor([0,1])}
    prediction,loss=graph.forward_inputs(values);graph.backward()
    gs=[p.grad.clone() for p in graph.graph.parameters()];graph.graph.zero_grad(set_to_none=True)
    direct,direct_loss=graph.native_forward_inputs(values);direct_loss.backward()
    assert torch.equal(loss,direct_loss) and all(torch.equal(prediction[k],direct[k]) for k in prediction)
    assert all(torch.allclose(a,p.grad,atol=1e-6,rtol=1e-5) for a,p in zip(gs,graph.graph.parameters()))
    assert all(ids[k]==graph.by_name[k].id for k in ids)
    saved=graph.save_state();graph.load_complete_state(saved)
    print(json.dumps(dict(passed=True,legacy_cases=3,initialization_state_exact=True,legacy_MHD_outputs_gradients_exact=True,
                          original_ids_and_levels_exact=True,imported_1D_3D_network=True,arbitrary_source_names=True,strict_reload=True)))

if __name__=='__main__':check()
