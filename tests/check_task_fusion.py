"""Task readout formulas, participant grouping, full MHD autograd and compatibility."""
import json
import torch
from radonbridge.task_fusion import TaskFusionHead, participant_tokens
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer, clip_task_gradients
from radonbridge.experiment import parameter_hash, selection_score


def check():
    torch.set_num_threads(3);torch.manual_seed(64)
    x=[torch.randn(4,6,3,4,dtype=torch.float64,requires_grad=True),torch.randn(4,6,2,3,4,dtype=torch.float64,requires_grad=True)]
    rng=torch.random.get_rng_state().clone()
    a=TaskFusionHead(pooling='mean',seed=34,channels=(6,6)).double()
    b=TaskFusionHead(pooling='gated_mil',seed=34,channels=(6,6)).double()
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,b.classifier.state_dict()[k]) for k,v in a.classifier.state_dict().items())
    expected=a.classifier(torch.cat([z.flatten(2).mean(-1).reshape(2,2,6).mean(1) for z in x],1))
    assert torch.allclose(a(*x),expected,atol=1e-12,rtol=1e-12)
    pooled=[]
    for z,pool in zip(x,b.pools):
        t=participant_tokens(z)
        score=((torch.tanh(t@pool.v.weight.T+pool.v.bias)*torch.sigmoid(t@pool.u.weight.T+pool.u.bias))@pool.w.weight.T)
        weights=torch.exp(score-score.max(1,keepdim=True).values);weights=weights/weights.sum(1,keepdim=True)
        pooled.append((weights*t).sum(1))
    assert torch.allclose(b(*x),b.classifier(torch.cat(pooled,1)),atol=1e-12,rtol=1e-12)
    # Within-participant spatial and eye permutations preserve bag output;
    # another participant's data does not alter this participant's prediction.
    permuted=[z.flip(0).flip(-1) for z in x]
    assert torch.allclose(b(*permuted),b(*x).flip(0),atol=1e-12,rtol=1e-12)
    changed=[z.detach().clone() for z in x];changed[1][2:]+=10
    assert torch.equal(b(*changed)[:1],b(*x)[:1])
    c=torch.randn(1,2,3,224,224);o=torch.randn(1,2,1,32,96,96);y=torch.tensor([1])
    old=PilotGraph(seed=3416);old.graph.eval();state=old.save_state();initial_hash=parameter_hash(old)
    with torch.no_grad():pred=old.forward(c,o,y)[0]
    native_ids={k:old.by_name[k].id for k in old.by_name if '_stage' in k}
    del old
    for pooling in ('mean','gated_mil'):
        cfg=dict(pooling=pooling,hidden_dimension=256,attention_dimension=128)
        g=PilotGraph(seed=3416,task_fusion=cfg);g.load_native_state(state);g.graph.eval()
        assert parameter_hash(g)==initial_hash
        out,loss=g.forward(c,o,y)
        assert all(torch.equal(out[k],pred[k]) for k in g.branches)
        assert all(g.by_name[k].id==v for k,v in native_ids.items())
        assert torch.equal(loss,sum(torch.nn.functional.cross_entropy(out[k],y) for k in out))
        params=list(g.graph.parameters());g.backward();gs=[p.grad.clone() for p in params]
        g.graph.zero_grad(set_to_none=True);native,nloss=g.native_forward(c,o,y);nloss.backward()
        for p,grad in zip(params,gs):assert torch.allclose(p.grad,grad,atol=2e-6,rtol=2e-5)
        assert torch.equal(out['fusion'],native['fusion'])
        saved=g.save_state()
        opt=configure_optimizer(g,dict(adapt_stages=[1,2,3,4],training_regime='full_finetune',backbone_lr=6e-5,head_lr=1e-4,bridge_lr=1e-4,weight_decay=.01))
        bn=[m.num_batches_tracked for m in g.graph.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        before=[z.clone() for z in bn];g.graph.train()
        for step in range(2):
            opt.zero_grad(set_to_none=True);g.forward(c,o,y);g.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(p.grad.norm()>0 for p in g.modules_by_name()['fusion_head'].parameters())
            assert set(clip_task_gradients(g,5.))=={'cfp','oct','fusion'};opt.step()
        assert all(a>b for a,b in zip(bn,before))
        for name,m in g.modules_by_name().items():m.load_state_dict(saved[name],strict=True)
        g.graph.eval()
        with torch.no_grad():assert torch.equal(g.forward(c,o,y)[0]['fusion'],out['fusion'])
        del g,opt
    assert selection_score({'tasks':{'fusion':{'macro_f1':.4}},'mean_task_macro_f1':.9},{'selection_metric':'fusion_macro_f1'})==.4
    assert selection_score({'mean_task_macro_f1':.9},{})==.9
    print(json.dumps(dict(passed=True,formula=True,participant_bag_identity=True,shared_classifier_initialization=True,random_state_preserved=True,native_outputs_and_nodes_preserved=True,MHD_native_gradients=True,BN_and_optimizer=True,checkpoint_reload=True,fusion_selection_separate=True)))
if __name__=='__main__':check()
