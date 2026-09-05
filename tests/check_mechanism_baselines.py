"""Formula, directed gradient, identity, MHD and strict legacy compatibility checks."""
import copy,json,math,tempfile
from pathlib import Path
import torch
from radonbridge.bridge import FeatureSpec,BridgeExchange,LinearMixer,attach_to_nodes
from radonbridge.baselines import MMTMExchange,AttentionExchange
from radonbridge.graph import MHDBuilder
from radonbridge.model import PilotGraph
from radonbridge.diagnostics import analyze_graph,release_forward_graph,state_hash
from check_integer_bridge import Stage,Objective


def check():
    torch.set_num_threads(3);torch.manual_seed(719)
    specs=[FeatureSpec('cfp',8,(3,4)),FeatureSpec('oct',6,(2,3,4))]
    x=[torch.randn(4,s.channels,*s.shape,dtype=torch.float64,requires_grad=True) for s in specs]
    for family,kwargs in [('mmtm',{'reduction_ratio':4}),('cross_attention',{'attention_dimension':8,'heads':4})]:
        cls=MMTMExchange if family=='mmtm' else AttentionExchange
        module=cls(specs,**kwargs).double()
        assert torch.equal(module(*x),torch.cat([v.flatten(1) for v in x],1))
        opt=torch.optim.AdamW(module.parameters(),lr=.002)
        for step in range(3):
            opt.zero_grad(set_to_none=True);module(*x).square().mean().backward()
            if step>0:assert all(p.grad is not None and p.grad.norm()>0 for p in module.parameters())
            opt.step()
        output=module(*x)
        if family=='mmtm':
            hidden=torch.relu(torch.nn.functional.linear(torch.cat([v.mean(tuple(range(2,v.ndim))) for v in x],1),module.squeeze.weight,module.squeeze.bias))
            expected=torch.cat([(v* (2*torch.sigmoid(torch.nn.functional.linear(hidden,l.weight,l.bias))).reshape(v.shape[0],v.shape[1],*([1]*(v.ndim-2)))).flatten(1) for v,l in zip(x,module.excite)],1)
        else:
            tokens=[torch.nn.functional.layer_norm(v.flatten(2).transpose(1,2),(v.shape[1],),n.weight,n.bias,n.eps) for n,v in zip(module.norms,x)]
            expected=[]
            for i,l in enumerate(module.directions):
                q=l.query(tokens[i]);k=l.key(tokens[1-i]);v=l.value(tokens[1-i]);pieces=[]
                for h in range(4):
                    sl=slice(h*2,(h+1)*2);pieces.append(torch.softmax(q[:,:,sl]@k[:,:,sl].transpose(1,2)/math.sqrt(2),-1)@v[:,:,sl])
                delta=l.output(torch.cat(pieces,-1)).transpose(1,2).reshape_as(x[i]);expected.append((x[i]+delta).flatten(1))
            expected=torch.cat(expected,1)
        assert torch.allclose(output,expected,atol=1e-12,rtol=1e-12)
        for dst in range(2):
            start=sum(module.lengths[:dst]);g=torch.autograd.grad(output[:,start:start+module.lengths[dst]].square().mean(),x[1-dst],retain_graph=True)[0]
            assert g.norm()>0
        restored=cls(specs,**kwargs).double();restored.load_state_dict(module.state_dict(),strict=True);assert torch.equal(restored(*x),output)
        b=MHDBuilder();inputs={};cuts=[];outputs=[]
        for i,s in enumerate(specs):
            inp=b.node(s.key);out=b.node(s.key+'_stage3');end=b.node(s.key+'_stage4')
            b.edge(s.key+'_stage3',Stage(),[inp],[out]);b.edge(s.key+'_stage4',Stage(),[out],[end]);inputs[s.key]=x[i];cuts.append(s.key+'_stage3');outputs.append(end)
        loss=b.node('loss');b.edge('objective',Objective(),outputs,[loss]);old={k:n.id for k,n in b.by_name.items()};samples=b.native_forward(inputs)
        attach_to_nodes(b,cuts,prefix='bridge_',samples=samples,family=family,**kwargs)
        graph=b.compile().double();ex=next(e.edge_operations[0].function for e in b.edges if e.name=='bridge_exchange')
        with torch.no_grad():
            for p in ex.parameters():p.add_(torch.randn_like(p)*.01)
        b.set_inputs(inputs);graph.forward(levels=b.forward_levels);graph.backward(levels=b.backward_levels);gs=[p.grad.clone() for p in graph.parameters()]
        graph.zero_grad(set_to_none=True);b.native_forward(inputs)['loss'].backward()
        for a,p in zip(gs,graph.parameters()):assert torch.allclose(a,p.grad,atol=1e-10,rtol=1e-10)
        assert all(b.by_name[k].id==n for k,n in old.items())
    for allowed in [[],[['oct','cfp']],[['cfp','oct']]]:
        m=LinearMixer([3,5],3,keys=['cfp','oct'],cross_edges=allowed).double();features=[torch.randn(2,w,7,dtype=torch.float64,requires_grad=True) for w in [3,5]]
        opt=torch.optim.AdamW(m.parameters(),lr=.01)
        for _ in range(3):
            opt.zero_grad(set_to_none=True);sum(v.square().sum()+v.sum() for v in m(*features)).backward();opt.step()
            assert (m.conv.weight*(1-m.mask)).abs().sum()==0
        out=m(*features)
        for dst,src in [(0,1),(1,0)]:
            grad=torch.autograd.grad(out[dst].square().sum(),features[src],retain_graph=True)[0]
            assert (grad.norm()>0)==([['cfp','oct'][src],['cfp','oct'][dst]] in allowed)
    # Default path construction and state names remain exactly identical.
    torch.manual_seed(77);a=BridgeExchange(specs,M=4,S=11,rho=.5).double()
    torch.manual_seed(77);b=BridgeExchange(specs,M=4,S=11,rho=.5,cross_edges=None).double()
    assert list(a.state_dict())==list(b.state_dict());b.load_state_dict(a.state_dict(),strict=True)
    assert torch.equal(a(*x),b(*x))
    # Full native networks, optimizer and read-only diagnostics, both baselines.
    from radonbridge.optimization import configure_optimizer
    c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([0])
    g0=PilotGraph(seed=3416);g0.graph.eval();native=g0.save_state()
    with torch.no_grad():expected=g0.forward(c,o,y)[0]
    del g0
    for family,kw in [('mmtm',{'reduction_ratio':4}),('cross_attention',{'attention_dimension':128,'heads':4})]:
        g=PilotGraph(seed=3416,bridge_configs=[dict(nodes=['cfp_stage3','oct_stage3'],family=family,**kw)]);g.load_native_state(native);g.graph.eval()
        with torch.no_grad():out=g.forward(c,o,y)[0]
        assert all(torch.equal(out[k],expected[k]) for k in out)
        opt=configure_optimizer(g,dict(adapt_stages=[1,2,3,4],training_regime='full_finetune',backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,weight_decay=.01))
        tracked=[v for n,v in g.graph.named_buffers() if n.endswith('num_batches_tracked')];old=[v.clone() for v in tracked];g.graph.train()
        for _ in range(2):opt.zero_grad(set_to_none=True);g.forward(c,o,y);g.backward();opt.step()
        assert all(a>b for a,b in zip(tracked,old))
        opt.zero_grad(set_to_none=True)
        class Data(torch.utils.data.Dataset):
            rows=[{'order':0},{'order':1}]
            def __len__(self):return 2
            def __getitem__(self,i):return c[0],o[0],torch.tensor(i),i
        h=state_hash(g);rng=torch.random.get_rng_state().clone();d=analyze_graph(g,Data(),{},batch=1,probe_count=2)
        assert state_hash(g)==h and torch.equal(rng,torch.random.get_rng_state())
        assert all(v['retained_energy_ratio'] is None for v in d['energy'].values())
        release_forward_graph(g);del g,opt
    print(json.dumps({'passed':True,'baseline_formulas':True,'native_MHD_gradients':True,'directed_masks':True,'legacy_default_strict_load':True,'native_identity_and_BN':True,'baseline_read_only_diagnostics':True}))

if __name__=='__main__':check()
