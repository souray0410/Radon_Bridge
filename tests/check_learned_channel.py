"""Linear encoder/decoder, initialization pairing, gradients and MHD integration."""
import json,tempfile
from pathlib import Path
import torch
from radonbridge.learned_channel import LearnedChannelCodec
from radonbridge.bridge import BridgeExchange,FeatureSpec
from radonbridge.svd_basis import save_random_basis,FixedChannelBasis,QR_VERSION
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer,clip_task_gradients
from radonbridge.diagnostics import learned_rowspace,read_only,state_hash,analyze_graph,release_forward_graph
from check_fixed_channel_basis import artifact
from check_integer_bridge import topology_check
from scripts.run_learned_channel_study import matrix

def check():
    torch.set_num_threads(3);torch.manual_seed(3416)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);ref,_=artifact(root,8,'a');qr=save_random_basis(ref,root/'qr')
        rng=torch.random.get_rng_state().clone();c=LearnedChannelCodec(8,4,'a',3416);assert torch.equal(rng,torch.random.get_rng_state())
        q=FixedChannelBasis(8,4,'a',qr,QR_VERSION);assert torch.equal(c.encoder.weight.squeeze(-1),q.q.T) and torch.equal(c.decoder.weight.squeeze(-1),q.q)
        narrow=LearnedChannelCodec(8,2,'a',3416);assert torch.equal(narrow.encoder.weight,c.encoder.weight[:2])
        assert c.encoder.weight.data_ptr()!=c.decoder.weight.data_ptr()
        assert not torch.equal(c.encoder.weight,LearnedChannelCodec(8,4,'a',3417).encoder.weight)
        for shape in [(4,5),(3,4,5)]:
            x=torch.randn(2,8,*shape,dtype=torch.float64)
            assert torch.allclose(c.encode(x),q.encode(x),atol=1e-12,rtol=1e-12)
            assert torch.allclose(c.decode(c.encode(x)),q.decode(q.encode(x)),atol=1e-12,rtol=1e-12)
            from radonbridge.projector import Projector
            p=Projector(shape,4,11)
            assert torch.allclose(p(c.encode(x)),c.encode(p(x).reshape(2,8,4,11)).reshape(2,16,11),atol=1e-12,rtol=1e-12)
        topo=[topology_check([2,3],compression='learned_channel',mode=mode) for mode in ['radon','self','scrambled','linear_resample']]
        specs=[FeatureSpec('a',8,(4,5)),FeatureSpec('b',6,(3,4,5))]
        bridge=BridgeExchange(specs,M={'a':4,'b':8},S=11,rho={'a':.5,'b':.25},compression='learned_channel').double()
        assert bridge.mixer.widths==(16,8)
        xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.float64) for s in specs]
        assert torch.equal(bridge(*xs),torch.cat([x.flatten(1) for x in xs],1))
        opt=torch.optim.AdamW(bridge.parameters(),lr=.001)
        bridge(*xs).square().mean().backward()
        assert all(p.grad.abs().sum()==0 for codec in bridge.channel_codecs for p in codec.parameters())
        assert bridge.mixer.conv.weight.grad.abs().sum()>0
        opt.step();opt.zero_grad(set_to_none=True)
        before={n:p.clone() for n,p in bridge.named_parameters()};bridge(*xs).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in bridge.parameters())
        opt.step();assert all(not torch.equal(before[n],p) for n,p in bridge.named_parameters())
        other=BridgeExchange(specs,M={'a':4,'b':8},S=11,rho={'a':.5,'b':.25},compression='learned_channel').double();other.load_state_dict(bridge.state_dict());assert torch.equal(other(*xs),bridge(*xs))
        row=learned_rowspace(c,'cpu');assert torch.allclose(row.q@row.q.T,q.q@q.q.T,atol=1e-12)
        with torch.no_grad():c.encoder.weight.mul_(3)
        row2=learned_rowspace(c,'cpu');assert torch.allclose(row.q@row.q.T,row2.q@row2.q.T,atol=1e-12)
        with torch.no_grad():c.encoder.weight.zero_()
        assert learned_rowspace(c,'cpu').q.shape==(8,0)
        assert len(matrix())==len(set(matrix()))==36
        g0=PilotGraph(seed=3416);native=g0.save_state();cfp=torch.randn(1,2,3,224,224);oct_=torch.randn(1,2,1,32,96,96);y=torch.tensor([0]);g0.graph.eval()
        with torch.no_grad():expected=g0.forward(cfp,oct_,y)[0]
        del g0
        cfg={'nodes':['cfp_stage3','oct_stage3'],'M':4,'S':11,'rho':.125,'mode':'radon','compression':'learned_channel'}
        g=PilotGraph(seed=3416,bridge_configs=[cfg]);g.load_native_state(native);g.graph.eval()
        with torch.no_grad():actual=g.forward(cfp,oct_,y)[0]
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
        opt=configure_optimizer(g,{'adapt_stages':[1,2,3,4],'training_regime':'full_finetune','backbone_lr':6e-5,'head_lr':1e-4,'bridge_lr':1e-4,'weight_decay':.01})
        assert {id(p) for group in opt.param_groups for p in group['params']}=={id(p) for p in g.graph.parameters()}
        tracked=[b for n,b in g.graph.named_buffers() if n.endswith('num_batches_tracked')];before=[b.clone() for b in tracked]
        ex=g.modules_by_name()['bridge_0_exchange'];codec_before=[p.clone() for p in ex.channel_codecs.parameters()];native_before=next(g.modules_by_name()['cfp_stage1'].parameters()).clone();g.graph.train()
        for _ in range(2):g.forward(cfp,oct_,y);g.backward();clip_task_gradients(g,5);opt.step();opt.zero_grad(set_to_none=True)
        assert all(after>prior for after,prior in zip(tracked,before)) and all(not torch.equal(a,b) for a,b in zip(codec_before,ex.channel_codecs.parameters()))
        assert not torch.equal(native_before,next(g.modules_by_name()['cfp_stage1'].parameters()))
        class FakeData(torch.utils.data.Dataset):
            rows=[{'order':0},{'order':1}]
            def __len__(self):return 2
            def __getitem__(self,i):return cfp[0],oct_[0],torch.tensor(i),i
        h=state_hash(g);rng=torch.random.get_rng_state().clone();diagnostic=analyze_graph(g,FakeData(),{},batch=1,probe_count=2)
        assert state_hash(g)==h and torch.equal(rng,torch.random.get_rng_state())
        assert len(diagnostic['gradient_groups'])==7
        assert all('encoded_energy_ratio' in v for v in diagnostic['energy'].values())
        assert diagnostic['gradient_groups']['bridge_0_exchange.mixer.conv']['cosine']==0
        release_forward_graph(g)
        print(json.dumps({'passed':True,'topology':topo,'paired_QR_initialization':True,'independent_encoder_decoder_update':True,'zero_start_predictions_exact':True,'checkpoint_reload_exact':True,'native_optimizer_and_BN_update':True,'read_only_full_graph_diagnostic':True,'rowspace_scaling_invariant':True,'matrix':36}))
if __name__=='__main__':check()
