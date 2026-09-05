"""QR controls, exact width export, nested MHD gradient accumulation and BN policy."""
import copy,json,tempfile
from pathlib import Path
import torch
from radonbridge.bridge import BridgeExchange,FeatureSpec
from radonbridge.svd_basis import save_random_basis
from radonbridge.nested_training import training_backward,bn_snapshot,restore_bn,merge_bn,set_width
from radonbridge.nested_export import width_state
from radonbridge.model import PilotGraph
from check_fixed_channel_basis import artifact
from check_integer_bridge import topology_check


def check():
    torch.set_num_threads(3);torch.manual_seed(3416)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);refs={k:artifact(root,c,k)[0] for k,c in [('a',8),('b',12)]};qr={k:save_random_basis(v,root/'qr') for k,v in refs.items()}
        specs=[FeatureSpec('a',8,(3,4)),FeatureSpec('b',12,(2,3,4))]
        xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64,requires_grad=True) for s in specs]
        for comp,base in [('learned_channel',None),('fixed_svd_channel',refs),('fixed_random_orthogonal_channel',qr)]:
            cfg={'nodes':['a','b'],'M':4,'S':11,'rho':.5,'mode':'radon','compression':comp,'nested_rhos':[.125,.25,.5]}
            if base:cfg['basis_files']=base
            kw={k:v for k,v in cfg.items() if k!='nodes'};ex=BridgeExchange(specs,**kw).double()
            for rho in cfg['nested_rhos']:
                ex.set_rho(rho);assert torch.equal(ex(*xs),torch.cat([x.flatten(1) for x in xs],1))
            with torch.no_grad():ex.mixer.conv.weight.normal_(std=.01)
            for rho in cfg['nested_rhos']:
                ex.set_rho(rho);output=ex(*xs)
                small_kw=copy.deepcopy(kw);small_kw.pop('nested_rhos');small_kw['rho']=rho
                small=BridgeExchange(specs,**small_kw).double()
                small.load_state_dict(width_state({'bridge_0_exchange':ex.state_dict()},{'bridges':[cfg]},rho)['bridge_0_exchange'],strict=True)
                assert torch.allclose(small(*xs),output,atol=1e-12,rtol=1e-12)
                for dst in range(2):
                    start=sum(ex.lengths[:dst]);grad=torch.autograd.grad(output[:,start:start+ex.lengths[dst]].sum(),xs[1-dst],retain_graph=True)[0];assert grad.norm()>0
                ex.zero_grad(set_to_none=True);output.square().mean().backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ex.parameters())
                # Inactive source-specific rows and columns receive exactly no gradient.
                ranks=[max(1,int(rho*s.channels))*4 for s in specs];offset=0
                for maximum,r in zip(ex.mixer.widths,ranks):
                    assert ex.mixer.conv.weight.grad[offset+r:offset+maximum].abs().sum()==0
                    assert ex.mixer.conv.weight.grad[:,offset+r:offset+maximum].abs().sum()==0
                    offset+=maximum
        topo_refs={k:artifact(root,c,k)[0] for k,c in [('network0_stage1',2),('network1_stage2',3)]}
        topo_qr={k:save_random_basis(v,root/'qr_topo') for k,v in topo_refs.items()}
        for mode in ['self','scrambled']:topology_check([2,3],compression='fixed_random_orthogonal_channel',basis_files=topo_qr,mode=mode)
        # QR and SVD scramble use identical saved seed-derived permutations.
        for mode in ['self','scrambled']:
            a=BridgeExchange(specs,M=4,S=11,rho=.5,mode=mode,compression='fixed_svd_channel',basis_files=refs)
            b=BridgeExchange(specs,M=4,S=11,rho=.5,mode=mode,compression='fixed_random_orthogonal_channel',basis_files=qr)
            if mode=='scrambled':assert all(torch.equal(x.permutation,y.permutation) for x,y in zip(a.projectors,b.projectors))
    cfg={'nodes':['cfp_stage3','oct_stage3'],'M':4,'S':11,'rho':.25,'mode':'radon','compression':'learned_channel','nested_rhos':[.0625,.125,.25]}
    g=PilotGraph(bridge_configs=[cfg],seed=3416);g.graph.train();ex=g.modules_by_name()['bridge_0_exchange']
    with torch.no_grad():ex.mixer.conv.weight.normal_(std=.001)
    c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([0])
    original=g.save_state();bn=bn_snapshot(g);g.graph.zero_grad(set_to_none=True)
    loss=training_backward(g,c,o,y);actual=[p.grad.clone() for p in g.graph.parameters()];after=bn_snapshot(g)
    assert all(torch.equal(after[m]['num_batches_tracked'],bn[m]['num_batches_tracked']+1) for m in bn)
    for k,m in g.modules_by_name().items():m.load_state_dict(original[k],strict=True)
    g.graph.zero_grad(set_to_none=True);updates=[];losses=[]
    for rho in cfg['nested_rhos']:
        restore_bn(bn);set_width(g,rho);g.modules_by_name()['task_loss_sum'].scale=1/3
        _,l=g.native_forward(c,o,y);l.backward();losses.append(float(l.detach()));updates.append(bn_snapshot(g))
    merge_bn(updates)
    assert abs(float(loss)-sum(losses))<1e-5
    for a,p in zip(actual,g.graph.parameters()):assert torch.allclose(a,p.grad,atol=3e-5,rtol=3e-4),float((a-p.grad).abs().max())
    for m,values in after.items():
        for k,v in values.items():assert torch.allclose(v,getattr(m,k))
    # After more than the zero-initialized step, every native and codec parameter remains trainable.
    opt=torch.optim.AdamW(g.graph.parameters(),lr=1e-4)
    before=[p.clone() for p in ex.channel_codecs.parameters()]
    for _ in range(2):opt.zero_grad(set_to_none=True);training_backward(g,c,o,y);opt.step()
    assert all(not torch.equal(a,b) for a,b in zip(before,ex.channel_codecs.parameters()))
    print(json.dumps({'passed':True,'QR_self_scrambled_MHD':True,'all_three_width_exports':True,'inactive_gradient_zero':True,'MHD_mean_gradient_matches_native_autograd':True,'BN_single_equal_update':True,'learned_codec_updates':True}))

if __name__=='__main__':check()
