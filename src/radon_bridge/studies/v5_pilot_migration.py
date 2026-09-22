"""Offline, lossless V4 PilotGraph conversion before numerical acceptance.

No runtime imports this module. Inputs require an independently audited source
model description; conversion alone never authorizes dispatch or reference reuse.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from radon_bridge.runtime.pilot_checkpoint import read, save


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def describe(graph,optimizer=None):
    aliases={}
    for name,param in graph.graph.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param),[]).append(name)
    return dict(node_ids=[[n.id,n.name] for n in sorted(graph.nodes,key=lambda n:n.id)],
                parameter_aliases=sorted(sorted(v) for v in aliases.values()),
                optimizer_class=None if optimizer is None else type(optimizer).__module__+'.'+type(optimizer).__qualname__,
                parameter_groups=None if optimizer is None else
                    [[aliases[id(p)] for p in group['params']] for group in optimizer.param_groups])


def _load_model(graph,model,kind,branch=None):
    expected=graph.save_state()
    if kind=='native_parent':
        if branch not in graph.definition.source_modules:raise ValueError('Unknown parent branch')
        expected={k:expected[k] for k in graph.definition.source_modules[branch]}
    if not isinstance(model,dict) or model.keys()!=expected.keys():raise ValueError('Module mapping is not an exact identity mapping')
    for module,values in expected.items():
        if model[module].keys()!=values.keys():raise ValueError('Parameter/buffer keys changed')
        for name,target in values.items():
            source=model[module][name]
            if isinstance(target,torch.Tensor):
                if not isinstance(source,torch.Tensor) or source.shape!=target.shape or source.dtype!=target.dtype:
                    raise ValueError('Parameter/buffer shape or dtype changed')
            elif not equal(source,target):raise ValueError('Unsupported changed extra module state')
    if kind=='native_parent':graph.load_native_state(model,branch)
    else:graph.load_complete_state(model)
    actual=graph.save_state()
    if not equal(model,{k:actual[k] for k in model}):raise ValueError('Strict load changed state or shared parameter aliases')


def convert(source,output,*,source_sha256,source_description,description_sha256,
            kind,graph,optimizer=None):
    """Convert one immutable state; completed-epoch resume is the only full kind."""
    source=Path(source);output=Path(output);metadata=Path(source_description)
    if kind not in ('selected','native_parent','resume'):raise ValueError('Unsupported V4 migration role')
    if sha(source)!=source_sha256 or sha(metadata)!=description_sha256:raise ValueError('Source hash changed')
    record=json.loads(metadata.read_text())
    if (record.get('schema')!='radon_v4_pilot_description_v1' or record.get('framework_api')!='V4'
            or record.get('source_verified') is not True or not record.get('source_files_sha256')
            or record.get('checkpoint_sha256')!=source_sha256 or record.get('kind')!=kind):
        raise ValueError('Independently verified V4 source description required')
    state=torch.load(source,map_location='cpu',weights_only=False)
    if not isinstance(state,dict) or {'format','framework_api','kind'} & state.keys():raise ValueError('Expected an unconverted V4 payload')
    target=describe(graph,optimizer)
    for field in ('node_ids','parameter_aliases','parameter_groups','optimizer_class'):
        if record.get(field)!=target[field]:raise ValueError('Source/target '+field+' changed')
    if kind=='native_parent':
        if state.get('training_stage')!='independent' or state.get('stop_reason')!='validation_plateau':
            raise ValueError('Unaccepted native parent')
    _load_model(graph,state['model'],kind,state.get('branch'))
    result=copy.deepcopy(state)
    if kind=='resume':
        required={'identity','configuration','epoch','monitor','history','optimizer','rng','selected'}
        if not required<=state.keys() or optimizer is None:raise ValueError('Incomplete full resume state')
        if (type(state['epoch']) is not int or state['epoch']<0
                or [r.get('epoch') for r in state['history']]!=list(range(1,state['epoch']+1))
                or not {'torch','cuda','numpy','python'}<=state['rng'].keys()):
            raise ValueError('Resume boundary, history or RNG is incomplete')
        selected=state['selected']
        if selected.get('configuration')!=state['configuration']:raise ValueError('Selected/resume configuration differs')
        _load_model(graph,selected['model'],'selected')
        _load_model(graph,state['model'],'resume')
        groups=state['optimizer']['param_groups']
        if len(groups)!=len(optimizer.param_groups):raise ValueError('Optimizer group count changed')
        flat=[pid for group in groups for pid in group['params']]
        if len(flat)!=len(set(flat)):raise ValueError('Repeated optimizer parameter identity')
        if set(state['optimizer']['state'])-set(flat):raise ValueError('Unmapped optimizer state')
        for source_group,dest_group in zip(groups,optimizer.param_groups):
            if len(source_group['params'])!=len(dest_group['params']):raise ValueError('Optimizer parameter count changed')
            for pid,param in zip(source_group['params'],dest_group['params']):
                for key,value in state['optimizer']['state'].get(pid,{}).items():
                    if torch.is_tensor(value) and key!='step' and (value.shape!=param.shape or value.dtype!=param.dtype):
                        raise ValueError('Optimizer tensor shape/dtype changed')
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        if not equal(state['optimizer'],optimizer.state_dict()):raise ValueError('Optimizer restore changed values')
        result['selected']=dict(selected,format='radon_pilot_state_v2',framework_api='V5',kind='selected')
    if sha(source)!=source_sha256 or sha(metadata)!=description_sha256:raise ValueError('Source changed during conversion')
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():raise FileExistsError(output)
        stage=Path(tempfile.mkdtemp(prefix=output.name+'.',dir=output.parent))
        try:
            save(result,stage/'checkpoint.pt',kind=kind)
            loaded=read(stage/'checkpoint.pt',kind=kind)
            for key in ('format','framework_api','kind'):loaded.pop(key)
            if kind=='resume':
                for key in ('format','framework_api','kind'):loaded['selected'].pop(key)
            if not equal(state,loaded):raise ValueError('Conversion changed original payload')
            receipt=dict(schema='radon_v5_pilot_conversion_v1',state='awaiting_numerical_replay',
                source_sha256=source_sha256,description_sha256=description_sha256,kind=kind,
                target_sha256=sha(stage/'checkpoint.pt'),original_payload_exact=True,
                source_framework_api='V4',execution_framework_api='V5',
                identity_mapping=True,references_migrated=False,dispatch_allowed=False,test_access=False)
            receipt['converter_sha256']=sha(__file__)
            lock_path=Path(__file__).resolve().parents[3]/'framework.lock.json'
            receipt['execution_framework']=json.loads(lock_path.read_text())
            (stage/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n')
            os.rename(stage,output)
        finally:
            if stage.exists():shutil.rmtree(stage)
    return receipt
