"""Existing-node topology, automatic heterogeneous cuts, and version-safe gradients."""
import json
import torch
from torch import nn
from radonbridge.graph import MHDBuilder
from radonbridge.bridge import attach_to_nodes,LinearMixer,FeatureSpec
from radonbridge.model import PilotGraph
from check_requirements import SquareLoss,exercise

class Stage(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(1.1))
    def forward(self,x): return torch.tanh(self.weight*x)

def check_cut(dims):
    b=MHDBuilder();x={};selected=[];outputs=[]
    for i,d in enumerate(dims):
        key=f'network{i}';previous=b.node(key)
        x[key]=torch.randn((2,i+2)+tuple(3+j%2 for j in range(d)),requires_grad=True)
        for depth in range(1,4):
            name=f'{key}_stage{depth}';n=b.node(name);b.edge(name,Stage(),[previous],[n]);previous=n
            if depth==1+i%2:selected.append(name)
        outputs.append(previous)
    loss=b.node('loss');b.edge('objective',SquareLoss(),outputs,[loss])
    native_nodes={k:n.id for k,n in b.by_name.items()};native_edges=list(b.steps)
    sample=b.native_forward(x);baseline=sample['loss'].detach()
    ids,meta=attach_to_nodes(b,selected,prefix='exchange_',samples=sample,upsilon=(1.,1.,.5))
    assert ids=={name:native_nodes[name] for name in selected}
    assert len(b.nodes)==len(native_nodes)+1
    assert all(b.by_name[k].id==v for k,v in native_nodes.items())
    assert {row[0]:(row[1],row[2]) for row in b.steps if row[0]<len(native_edges)}=={e:(h,t) for e,h,t in native_edges}
    g=b.compile('cpu');b.set_inputs(x);g.forward(levels=b.forward_levels)
    assert torch.equal(b.by_name['loss'].feature_message.current_state,baseline)
    exchange_id=len(native_edges);returns=range(exchange_id+1,len(b.edges))
    levels=[b.forward_edge_levels[i] for i in returns]
    assert len(set(levels))==1 and levels[0]==b.forward_edge_levels[exchange_id]+1
    for m in g.modules():
        if isinstance(m,LinearMixer):nn.init.normal_(m.conv.weight,std=.02)
    b.set_inputs(x);g.forward(levels=b.forward_levels);g.backward(levels=b.backward_levels)
    saved=[p.grad.clone() for p in g.parameters()];g.zero_grad(set_to_none=True)
    b.native_forward(x)['loss'].backward()
    for a,p in zip(saved,g.parameters()):assert torch.allclose(a,p.grad,atol=2e-6,rtol=1e-4)
    for _ in range(2):
        fresh={k:torch.randn_like(v,requires_grad=True) for k,v in x.items()};g.zero_grad(set_to_none=True)
        expected=b.native_forward(fresh)['loss'].detach();b.set_inputs(fresh);g.forward(levels=b.forward_levels)
        assert torch.equal(expected,b.by_name['loss'].feature_message.current_state)
        g.backward(levels=b.backward_levels)
    return {'dimensions':dims,'participants':len(dims),'native_nodes_preserved':True,'native_edges_preserved':True,'single_communication_node':True,'return_level':levels[0],'automatic_mixed_depth_cut':True,'mhd_native_gradients_match':True,'repeated_forward_backward':True}

def main():
    torch.set_num_threads(2);torch.manual_seed(991)
    reports=[check_cut(dims) for dims in [(2,3),(1,2,3),(1,2,3,4),(2,2,3,1,2)]]
    exercise([FeatureSpec('a',2,(3,4),(3,)),FeatureSpec('b',3,(3,4,3),(3,3))],repeats=3)
    base=PilotGraph(loss_reduction='sum');joined=PilotGraph('radon',bridge_stages=(2,3),loss_reduction='sum',upsilon=(1.,1.,.125))
    assert {k:n.id for k,n in base.by_name.items()}=={k:joined.by_name[k].id for k in base.by_name}
    assert len(joined.nodes)==len(base.nodes)+2
    assert [(i,h,t) for i,h,t in joined.steps if i<len(base.edges)]==base.steps
    assert not any(name.endswith('_updated') for name in joined.by_name)
    c=torch.randn(1,2,3,96,96);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
    a,_=base.forward(c,o,y);z,_=joined.forward(c,o,y);assert all(torch.equal(a[k],z[k]) for k in a)
    joined.backward()
    # One causal frontier per network; reject an ancestor/descendant pair.
    b=MHDBuilder();x=b.node('input');n=b.node('early');q=b.node('late');b.edge('early',Stage(),[x],[n]);b.edge('late',Stage(),[n],[q])
    samples=b.native_forward({'input':torch.ones(1,2,3)})
    try:attach_to_nodes(b,['early','late'],prefix='invalid_',samples=samples)
    except ValueError:pass
    else:raise AssertionError('Invalid causal cut accepted')
    print(json.dumps({'automatic_cases':reports,'repeated_groups':3,'pilot_native_ids_and_edges_preserved':True,'zero_bridge_exact':True,'causally_nested_cut_rejected':True}))
if __name__=='__main__':main()
