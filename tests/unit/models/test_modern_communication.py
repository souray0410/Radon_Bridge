import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from radon_bridge.models.graph import MHDBuilder
from radon_bridge.models.modern_communication import ModernMMTMHost
from radon_bridge.methods.baselines import AuthorMMTMExchange
from radon_bridge.methods.operator import FeatureSpec


def parent(dims):
    b = MHDBuilder()
    current = b.node('input')
    endpoints = {'input': current}
    conv, bn = (nn.Conv2d, nn.BatchNorm2d) if dims == 2 else (nn.Conv3d, nn.BatchNorm3d)
    channels = 1
    for stage in range(1, 5):
        name = 'stage' + str(stage)
        target = b.node(name)
        b.edge(name, nn.Sequential(conv(channels, 4, 1), bn(4), nn.ReLU()), [current], [target])
        current, channels = target, 4
        endpoints[name] = target
    features = b.node('features')
    pool = nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool3d(1)
    b.edge('features', nn.Sequential(pool, nn.Flatten(1)), [current], [features])
    logits = b.node('logits')
    b.edge('logits', nn.Linear(4, 2), [features], [logits])
    endpoints.update(features=features, logits=logits)
    g = SimpleNamespace(config=SimpleNamespace(views=1, spatial_dims=dims),
                        nodes=b.nodes, edges=b.edges, endpoint_nodes=endpoints,
                        _definitions=[(eid, heads, tails[0]) for eid, heads, tails in b.steps])
    return SimpleNamespace(graph=g)


def test_mmtm_author_formula_and_input_parameter_gradients():
    torch.manual_seed(17)
    specs = [FeatureSpec('a', 3, (2, 4)), FeatureSpec('b', 5, (2, 3, 2))]
    actual = AuthorMMTMExchange(specs, 4).double()
    reference = copy.deepcopy(actual)
    xs = [torch.randn(2, s.channels, *s.shape, dtype=torch.float64, requires_grad=True) for s in specs]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    shared = torch.relu(reference.squeeze(torch.cat([y.flatten(2).mean(-1) for y in ys], 1)))
    expected = torch.cat([(y * reference.excite[i](shared).sigmoid().view(y.shape[:2] + (1,) * (y.ndim-2))).flatten(1)
                          for i, y in enumerate(ys)], 1)
    output = actual(*xs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    cotangent = torch.randn_like(output)
    output.backward(cotangent)
    expected.backward(cotangent)
    for a, b in zip(xs + list(actual.parameters()), ys + list(reference.parameters())):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('frozen', [False, True])
def test_complete_mmtm_v5_native_continuous_updates_and_restore(frozen):
    torch.set_num_threads(2)
    torch.manual_seed(3416)
    parents = {'cfp': parent(2), 'oct': parent(3)}
    shapes = {'cfp': (1, 4, 4), 'oct': (1, 3, 4, 4)}
    model = ModernMMTMHost(parents, shapes, frozen=frozen).train()
    reference = ModernMMTMHost(parents, shapes, frozen=frozen).train()
    reference.load_state_dict(model.state_dict(), strict=True)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    original_modules = {name: copy.deepcopy(module.state_dict())
                        for name, module in model.task.modules_by_name().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    refopt = torch.optim.Adam(reference.parameters(), lr=1e-3)
    for step in range(3):
        values = {'cfp': torch.randn(3, *shapes['cfp']), 'oct': torch.randn(3, *shapes['oct']),
                  'counts': [1, 2], 'label': torch.tensor([0, 1])}
        left = {k: v.detach().clone().requires_grad_() if k in ('cfp', 'oct') else v for k, v in values.items()}
        right = {k: v.detach().clone().requires_grad_() if k in ('cfp', 'oct') else v for k, v in values.items()}
        opt.zero_grad(set_to_none=True); refopt.zero_grad(set_to_none=True)
        outputs, loss = model(left)
        targets, refloss = reference.native_forward(right)
        for key in outputs:
            torch.testing.assert_close(outputs[key], targets[key], rtol=1e-5, atol=1e-6)
        model.backward(); refloss.backward()
        for key in ('cfp', 'oct'):
            torch.testing.assert_close(left[key].grad, right[key].grad, rtol=1e-5, atol=1e-6)
        for (name, a), (refname, b) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == refname
            if a.requires_grad:
                assert a.grad is not None, name
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
            else:
                assert a.grad is None and b.grad is None
        opt.step(); refopt.step()
        for key, val in model.state_dict().items():
            torch.testing.assert_close(val, reference.state_dict()[key], rtol=1e-5, atol=1e-6)
        if step == 0:
            restored = ModernMMTMHost(parents, shapes, frozen=frozen).train()
            restored.load_state_dict(model.state_dict(), strict=True)
            newopt = torch.optim.Adam(restored.parameters(), lr=1e-3)
            newopt.load_state_dict(copy.deepcopy(opt.state_dict()))
            model, opt = restored, newopt
    if frozen:
        # Parameter/buffer names belong to actual registered graph operations.
        modules = model.task.modules_by_name()
        for name, module in modules.items():
            if not name.startswith('bridge_'):
                assert not module.training
        changed = [key for key, val in model.state_dict().items() if not torch.equal(val, original[key])]
        assert changed
        for name, module in modules.items():
            if name.startswith(('cfp_', 'oct_')):
                for key, val in module.state_dict().items():
                    torch.testing.assert_close(val, original_modules[name][key], rtol=0, atol=0)


def test_reject_missing_corresponding_site_and_unpaired_eyes():
    parents = {'cfp': parent(2), 'oct': parent(3)}
    shapes = {'cfp': (1, 4, 4), 'oct': (1, 3, 4, 4)}
    with pytest.raises(ValueError, match='endpoint'):
        ModernMMTMHost(parents, shapes, sites=('unknown',))
    model = ModernMMTMHost(parents, shapes)
    with pytest.raises(ValueError, match='Paired'):
        model({'cfp': torch.zeros(2, *shapes['cfp']), 'oct': torch.zeros(1, *shapes['oct']),
               'counts': [1], 'label': torch.tensor([0])})


@pytest.mark.parametrize('frozen', [False, True])
def test_standard_training_groups_and_node_identity(frozen):
    parents = {'cfp': parent(2), 'oct': parent(3)}
    shapes = {'cfp': (1, 4, 4), 'oct': (1, 3, 4, 4)}
    model = ModernMMTMHost(parents, shapes, frozen=frozen)
    groups = model.groups(1e-5, 1e-4, 1e-3)
    params = [p for group in groups for p in group['params']]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    assert {g['source'] for g in groups} == ({'bridge'} if frozen else {'cfp', 'oct', 'bridge'})
    for group in groups:
        if group['source'] == 'bridge':
            assert group['lr'] == 1e-3
    for p in params:
        p.grad = torch.ones_like(p)
    model.clip(.1)
    assert all(torch.isfinite(p.grad).all() for p in params)
    assert set(model.parent_node_map) == {'cfp', 'oct'}
    assert any(name == 'joint_logits' for _, name in model.node_identity())


@pytest.mark.parametrize('mode', ['radon', 'linear_resample'])
def test_multistage_matched_addition_native_gradients_and_restore(mode):
    torch.manual_seed(3416)
    parents = {'cfp': parent(2), 'oct': parent(3)}
    shapes = {'cfp': (1, 4, 4), 'oct': (1, 3, 4, 4)}
    host = ModernMMTMHost(parents, shapes, frozen=True)
    additions = {site: dict(M=2, S=4, rho=1, mode=mode,
                           compression='factorized_projected', bottleneck_rank=3)
                 for site in ('stage2', 'stage3', 'stage4')}
    actual = ModernMMTMHost(parents, shapes, frozen=True, additions=additions).train()
    actual.load_matched_host(host.task.save_state())
    reference = ModernMMTMHost(parents, shapes, frozen=True, additions=additions).train()
    reference.load_state_dict(actual.state_dict(), strict=True)
    original_parents = {n: copy.deepcopy(m.state_dict()) for n,m in actual.task.modules_by_name().items()
                        if n.startswith(('cfp_', 'oct_'))}
    opt = torch.optim.Adam(actual.groups(1e-5, 1e-4, 1e-3))
    refopt = torch.optim.Adam(reference.groups(1e-5, 1e-4, 1e-3))
    for step in range(3):
        batch = dict(cfp=torch.randn(3,*shapes['cfp']), oct=torch.randn(3,*shapes['oct']),
                     counts=[1,2], label=torch.tensor([0,1]))
        left = {k:v.clone().requires_grad_() if k in ('cfp','oct') else v for k,v in batch.items()}
        right = {k:v.clone().requires_grad_() if k in ('cfp','oct') else v for k,v in batch.items()}
        opt.zero_grad(set_to_none=True); refopt.zero_grad(set_to_none=True)
        out, loss = actual(left); expected, ref_loss = reference.native_forward(right)
        torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
        for key in out: torch.testing.assert_close(out[key], expected[key], rtol=1e-5, atol=1e-6)
        actual.backward(); ref_loss.backward()
        for key in ('cfp','oct'): torch.testing.assert_close(left[key].grad, right[key].grad, rtol=1e-5, atol=1e-6)
        for (n,a),(rn,b) in zip(actual.named_parameters(),reference.named_parameters()):
            assert n==rn
            if a.requires_grad:
                assert a.grad is not None, n
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
        opt.step(); refopt.step()
        for n,t in actual.state_dict().items():
            torch.testing.assert_close(t, reference.state_dict()[n],rtol=1e-5,atol=1e-6)
        if step==0:
            restored=ModernMMTMHost(parents,shapes,frozen=True,additions=additions).train()
            restored.load_state_dict(actual.state_dict(),strict=True)
            fresh=torch.optim.Adam(restored.groups(1e-5,1e-4,1e-3))
            fresh.load_state_dict(copy.deepcopy(opt.state_dict()))
            actual,opt=restored,fresh
    for n,buffers in original_parents.items():
        for key,t in buffers.items():
            torch.testing.assert_close(actual.task.modules_by_name()[n].state_dict()[key],t,rtol=0,atol=0)


def test_matched_host_transfer_rejects_missing_modules_before_mutation():
    parents={'cfp':parent(2),'oct':parent(3)}
    shapes={'cfp':(1,4,4),'oct':(1,3,4,4)}
    host=ModernMMTMHost(parents,shapes)
    augmented=ModernMMTMHost(parents,shapes,additions={'stage3':dict(M=2,S=4,rho=1,mode='radon')})
    state=host.task.save_state(); state.pop(next(iter(state)))
    before=copy.deepcopy(augmented.state_dict())
    with pytest.raises(ValueError,match='Selected host modules'):
        augmented.load_matched_host(state)
    for n,t in before.items():torch.testing.assert_close(t,augmented.state_dict()[n],rtol=0,atol=0)


def test_modern_selection_uses_joint_classifier_not_mean_branch_metric():
    from radon_bridge.evaluation.paired_native import selection_score
    result={'cfp':{'macro_f1':.8},'oct':{'macro_f1':.4},'joint':{'macro_f1':.7},'mean_macro_f1':.6}
    modern=SimpleNamespace(selection_output='joint',task=SimpleNamespace(output_names=('cfp','oct','joint')))
    assert selection_score(modern,result)==.7
    assert selection_score(SimpleNamespace(),result)==.6
    with pytest.raises(ValueError):selection_score(SimpleNamespace(selection_output='joint',task=SimpleNamespace(output_names=('cfp','oct'))),result)


def test_full_joint_prediction_saved_and_branch_mean_preserved(tmp_path,monkeypatch):
    import sys
    import numpy as np
    from radon_bridge.evaluation.paired_native import evaluate
    # Isolate this evaluator from unrelated native dataset initialization.
    monkeypatch.setitem(sys.modules,'mhd_models.workflows.native',SimpleNamespace(metrics=lambda y,p:{'macro_f1':float((p.argmax(1)==y).mean())}))
    class Model:
        training=True
        task=SimpleNamespace(output_names=('cfp','oct','joint'))
        def train(self,mode=True):self.training=mode;return self
        def eval(self):return self.train(False)
        def __call__(self,b):
            a=torch.tensor([[5.,0.],[5.,0.]])
            z=torch.tensor([[3.,0.],[0.,8.]])
            return {'cfp':a,'oct':z,'joint':(a+z)/2},None
    path=tmp_path/'prediction.npz'
    value=evaluate(Model(),[{'participant_id':['one','two'],'label':torch.tensor([0,1])}],'cpu',path)
    assert value['mean_macro_f1']==.75 and value['joint']['macro_f1']==1.
    with np.load(path,allow_pickle=False) as data:assert set(data.files)=={'participant_ids','labels','cfp','oct','joint'}
