"""Protocol arithmetic, legacy equality, new kernels, parallel MHD/optimizer checks."""
import argparse,json,subprocess,tempfile,types
from pathlib import Path
import torch
from radonbridge.bridge import BridgeExchange,FeatureSpec
from radonbridge.svd_basis import save_basis
from radonbridge.geometry_study import catalog,cost,matched_rank,fingerprint,configuration
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer,clip_task_gradients
from radonbridge.diagnostics import read_only,release_forward_graph

def basis(tmp,C,key):
    f=torch.randn(C,3*C,generator=torch.Generator().manual_seed(729),dtype=torch.float64)
    return save_basis(f@f.T,f.shape[1],tmp,key,3416,{'synthetic':True})

def main(output):
    torch.set_num_threads(3);report={}
    cat=catalog();assert len(cat['structures'])==193 and cat['geometry_positions']==2316
    assert len({s['id'] for s in cat['structures']})==193
    assert sum(c['radon_solution']['feasible'] for c in cat['compute_cells'])==38
    assert sum('compute_ordinary_id' in c for c in cat['compute_cells'])==31
    for cell in cat['compute_cells']:
        for mode in ('radon','linear_resample'):
            best=matched_rank(cell['M'],cell['S'],cell['k'],mode,cell['target_flops'])
            assert 8<=best['r']<=64
    report['catalog']=dict(positions=2316,structures=193,equal_parameter_positions=1296,baseline_positions=108)
    legacy=types.ModuleType('radonbridge._geometry_legacy');legacy.__package__='radonbridge'
    exec(compile(subprocess.check_output(['git','show','18c804a:radonbridge/bridge.py'],text=True),'legacy_bridge','exec'),legacy.__dict__)
    with tempfile.TemporaryDirectory() as tmp:
        specs=[FeatureSpec('a',6,(3,4)),FeatureSpec('b',6,(2,3,4))]
        refs={s.key:basis(tmp,6,s.key) for s in specs}
        features=[torch.randn(2,s.channels,*s.shape,requires_grad=True) for s in specs]
        for compression in ('learned_projected','fixed_svd_channel'):
            kw=dict(M=4,S=7,rho=.5,compression=compression)
            if compression=='fixed_svd_channel':kw['basis_files']=refs
            torch.manual_seed(99);old=legacy.BridgeExchange(specs,**kw).float()
            torch.manual_seed(99);new=BridgeExchange(specs,**kw).float()
            new.load_state_dict(old.state_dict(),strict=True)
            with torch.no_grad():old.mixer.conv.weight.normal_(std=.01)
            new.load_state_dict(old.state_dict(),strict=True)
            assert torch.equal(old(*features),new(*features))
            ga=torch.autograd.grad(old(*features).square().sum(),features)
            gb=torch.autograd.grad(new(*features).square().sum(),features)
            assert all(torch.equal(a,b) for a,b in zip(ga,gb))
        for k in (1,3,5):
            for mode in ('radon','linear_resample'):
                b=BridgeExchange(specs,M=4,S=7,rho=.5,r=3,h=12,kernel_size=k,compression='fixed_svd_channel',basis_files=refs,mode=mode).float()
                assert b.mixer.conv.weight.numel()==k*24**2
                assert torch.equal(b(*features),torch.cat([x.flatten(1) for x in features],1))
                with torch.no_grad():b.mixer.conv.weight.normal_(std=.01)
                out=b(*features);grads=torch.autograd.grad(out[:,:72].square().sum(),features)
                assert all(g.norm()>0 for g in grads)
                assert set(dict(b.named_parameters()))=={'mixer.conv.weight'}
        refs={s:basis(Path(tmp)/'full',256,s) for s in ('cfp_stage3','oct_stage3')}
        for family in ('mmtm','cross_attention'):
            host=dict(nodes=['cfp_stage3','oct_stage3'],family=family)
            host.update({'reduction_ratio':8} if family=='mmtm' else {'attention_dimension':128,'heads':4})
            g=PilotGraph(seed=3416,bridge_configs=[host]);g.graph.eval()
            with torch.no_grad():
                for p in g.modules_by_name()['bridge_0_exchange'].parameters():p.add_(torch.randn_like(p)*.003)
            saved=g.save_state()
            new=dict(nodes=host['nodes'],M=16,S=32,kernel_size=5,r=8,h=128,rho=8/256,mode='radon',compression='fixed_svd_channel',basis_files=refs,parallel_to=0)
            joined=PilotGraph(seed=3416,bridge_configs=[host,new]);joined.load_complete_state(saved,allow_new_bridge=True);joined.graph.eval()
            c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
            with torch.no_grad():
                a,_=g.forward(c,o,y);b,_=joined.forward(c,o,y)
                assert all(torch.equal(a[k],b[k]) for k in a)
            assert all(g.by_name[k].id==joined.by_name[k].id for k in g.by_name)
            recipe=dict(adapt_stages=[1,2,3,4],backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,weight_decay=.01,training_regime='full_finetune')
            opt=configure_optimizer(joined,recipe)
            for step in range(2):
                opt.zero_grad(set_to_none=True);joined.graph.train();_,loss=joined.forward(c,o,y);joined.backward()
                norms=clip_task_gradients(joined,5.);assert {'bridge_0','bridge_1','cfp','oct'}<=set(norms)
                opt.step();release_forward_graph(joined)
            joined.graph.eval();opt.zero_grad(set_to_none=True)
            _,loss=joined.forward(c,o,y);joined.backward()
            grads={n:[p.grad.clone() if p.grad is not None else None for p in m.parameters()] for n,m in joined.modules_by_name().items()}
            opt.zero_grad(set_to_none=True);_,native_loss=joined.native_forward(c,o,y);native_loss.backward()
            for n,m in joined.modules_by_name().items():
                for a,p in zip(grads[n],m.parameters()):
                    assert (a is None)==(p.grad is None)
                    if a is not None:assert torch.allclose(a,p.grad,atol=3e-5,rtol=3e-4),n
            with read_only(joined),torch.no_grad():joined.forward(c,o,y)
            del g,joined,opt
    report.update(passed=True,legacy_output_gradient_strict_load=True,kernels=[1,3,5],parallel_initial_host_identity=True,
                  parallel_mhd_native_gradients=True,original_node_ids=True,separate_clipping=True,read_only_state=True)
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
