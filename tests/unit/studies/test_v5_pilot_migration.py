import copy
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from radon_bridge.studies.v5_pilot_migration import convert,describe,sha,equal
from radon_bridge.runtime.pilot_checkpoint import read


class Graph:
    def __init__(self):
        self.graph=torch.nn.ModuleDict({'cfp_head':torch.nn.Linear(3,2)})
        self.nodes=[SimpleNamespace(id=0,name='input'),SimpleNamespace(id=1,name='logits')]
        self.definition=SimpleNamespace(source_modules={'cfp':['cfp_head']})
    def save_state(self):
        return {n:copy.deepcopy(m.state_dict()) for n,m in self.graph.items()}
    def load_complete_state(self,state):
        assert state.keys()==self.graph.keys()
        for name,values in state.items():self.graph[name].load_state_dict(values,strict=True)
    def load_native_state(self,state,branch):self.load_complete_state(state)


def fixture(tmp_path,kind='resume',alter=None):
    torch.manual_seed(43)
    graph=Graph();opt=torch.optim.AdamW(graph.graph.parameters(),lr=.01)
    graph.graph['cfp_head'](torch.ones(2,3)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    cfg=dict(seed=43,bridges=[])
    selected=dict(model=graph.save_state(),configuration=cfg,epoch=0,metrics={'f1':.5})
    state=copy.deepcopy(selected)
    if kind=='resume':
        state=dict(model=graph.save_state(),configuration=cfg,identity='original-run',epoch=1,
            history=[{'epoch':1}],monitor={'best_epoch':0},optimizer=opt.state_dict(),
            rng=dict(torch=torch.get_rng_state(),cuda=torch.zeros(1,dtype=torch.uint8),
                numpy=np.random.get_state(),python=random.getstate()),selected=selected)
    elif kind=='native_parent':
        state=dict(model=graph.save_state(),branch='cfp',training_stage='independent',seed=43,stop_reason='validation_plateau')
    if alter:alter(state)
    source=tmp_path/'old.pt';torch.save(state,source)
    target=Graph();optimizer=torch.optim.AdamW(target.graph.parameters(),lr=.01) if kind=='resume' else None
    description=dict(schema='radon_v4_pilot_description_v1',framework_api='V4',source_verified=True,
        source_files_sha256={'fixture':'a'*64},checkpoint_sha256=sha(source),kind=kind,
        **describe(graph,opt if kind=='resume' else None))
    meta=tmp_path/'description.json';meta.write_text(json.dumps(description))
    kwargs=dict(source_sha256=sha(source),source_description=meta,description_sha256=sha(meta),kind=kind,
                graph=target,optimizer=optimizer)
    return source,state,kwargs


@pytest.mark.parametrize('kind',['selected','native_parent','resume'])
def test_conversion_preserves_original_payload_and_current_strict_load(tmp_path,kind):
    source,state,kwargs=fixture(tmp_path,kind)
    before=source.read_bytes();out=tmp_path/'new'
    receipt=convert(source,out,**kwargs)
    assert source.read_bytes()==before
    assert receipt['dispatch_allowed'] is False and receipt['references_migrated'] is False
    loaded=read(out/'checkpoint.pt',kind=kind)
    for key in ('kind','framework_api','format'):loaded.pop(key)
    if kind=='resume':
        for key in ('kind','framework_api','format'):loaded['selected'].pop(key)
    assert equal(loaded,state)
    with pytest.raises(FileExistsError):convert(source,out,**kwargs)


@pytest.mark.parametrize('field',['node_ids','parameter_aliases','parameter_groups','optimizer_class'])
def test_conversion_rejects_changed_source_mapping_before_publication(tmp_path,field):
    source,_,kwargs=fixture(tmp_path)
    p=kwargs['source_description'];description=json.loads(p.read_text());description[field]=None
    p.write_text(json.dumps(description));kwargs['description_sha256']=sha(p)
    with pytest.raises(ValueError,match=field):convert(source,tmp_path/'new',**kwargs)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('alter',[
    lambda s:s.pop('rng'),
    lambda s:s['history'].clear(),
    lambda s:s['model']['cfp_head'].__setitem__('weight',s['model']['cfp_head']['weight'].double()),
    lambda s:s['selected'].__setitem__('configuration',{'seed':999}),
])
def test_conversion_rejects_incomplete_or_incompatible_state(tmp_path,alter):
    source,_,kwargs=fixture(tmp_path,alter=alter)
    with pytest.raises(ValueError):convert(source,tmp_path/'new',**kwargs)
    assert not (tmp_path/'new').exists()


def test_conversion_does_not_accept_changed_source_file(tmp_path):
    source,_,kwargs=fixture(tmp_path)
    source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash changed'):convert(source,tmp_path/'new',**kwargs)
