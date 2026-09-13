"""Compose complete native MHD graphs with observed-eye pooling and two heads."""
import copy
import torch
from torch import nn
from radon_bridge.models.graph import MHDBuilder
from radon_bridge.models.interface import NativeNetwork
from radon_bridge.models.networks import TaskLoss
from radon_bridge.models.task import MHDTaskGraph


class ParticipantPool(nn.Module):
    def __init__(self):
        super().__init__(); self.counts=(1,)
    def forward(self, x):
        if sum(self.counts)!=len(x) or any(n not in (1,2) for n in self.counts):
            raise ValueError('Observed-eye ownership mismatch')
        return torch.stack([part.mean(0) for part in x.split(self.counts)])


class PairLoss(TaskLoss):
    weights=(1.,1.)
    def forward(self,*losses):
        return sum(w*loss for w,loss in zip(self.weights,losses))*self.scale


def definition(parents, shapes):
    if set(parents)!={'cfp','oct'} or set(shapes)!=set(parents):raise ValueError('Two explicit source models required')
    builder=MHDBuilder();node,edge=builder.node,builder.edge
    pools={};ownership={};mapping={};predictions={};losses=[]
    target=node('target')
    for role in ('cfp','oct'):
        original=parents[role].graph
        if original.config.views!=1:raise ValueError('Native models must operate on individual observed eyes')
        if original.config.spatial_dims!=(2 if role=='cfp' else 3):raise ValueError('CFP2D/OCT3D required')
        local={}
        for old in sorted(original.nodes,key=lambda n:n.id):
            local[old.id]=node(role+'_'+old.name)
        mapping[role]={str(old.id):local[old.id] for old in original.nodes}
        for alias,old_id in original.endpoint_nodes.items():
            builder.by_name[role+'_'+alias]=builder.nodes[local[old_id]]
        pool=ParticipantPool();pools[role]=pool
        pooled=node(role+'_participant');feature=original.endpoint_nodes['features']
        ownership[role]=[]
        by_id={e.id:e for e in original.edges}
        for eid,heads,tail in original._definitions:
            source=by_id[eid];name=role+'_'+source.name
            edge(name,copy.deepcopy(source.edge_operations[0].function),
                 [pooled if h==feature else local[h] for h in heads],[local[tail]])
            ownership[role].append(name)
            if tail==feature:
                edge(role+'_participant_pool',pool,[local[feature]],[pooled]);ownership[role].append(role+'_participant_pool')
        predictions[role]=role+'_logits'
        loss=node(role+'_loss');edge(role+'_criterion',nn.CrossEntropyLoss(),[local[original.endpoint_nodes['logits']],target],[loss])
        losses.append(loss)
    loss=node('loss');edge('task_loss_sum',PairLoss(),losses,[loss])
    native_names=tuple(e.name for e in builder.edges)
    inputs={role+'_input':torch.zeros((1,*shapes[role])) for role in ('cfp','oct')}
    inputs['target']=torch.zeros(1,dtype=torch.long)
    return NativeNetwork(builder=builder,input_names=tuple(inputs),prediction_nodes=predictions,loss_node='loss',
        branches=('cfp','oct'),source_modules={k:tuple(v) for k,v in ownership.items()},
        native_checkpoint_modules=native_names,loss_scaler_edge='task_loss_sum',probe_inputs=inputs,
        metadata={'head_mode':'separate','parent_node_map':mapping}),pools


class NativePair(nn.Module):
    def __init__(self, parents, shapes, bridges=(), *, device='cpu', frozen=False):
        super().__init__()
        native,self.pools=definition(parents,shapes)
        self.task=MHDTaskGraph(native,bridge_configs=bridges,device=device)
        self.graph=self.task.graph
        self.parent_node_map=native.metadata['parent_node_map']
        self.frozen=frozen
        if frozen:
            for name,module in self.task.modules_by_name().items():
                if not name.startswith('bridge_'):
                    for p in module.parameters():p.requires_grad_(False)
        self.train(False)
    def train(self, mode=True):
        super().train(mode)
        if getattr(self,'frozen',False):
            for name,module in self.task.modules_by_name().items():
                if not name.startswith('bridge_'):module.eval()
        return self
    def forward(self, batch, loss_scale=1.):
        for p in self.pools.values():p.counts=tuple(batch['counts'])
        return self.task.forward_inputs({'cfp_input':batch['cfp'],'oct_input':batch['oct'],'target':batch['label']},loss_scale)
    def backward(self):self.task.backward()
    def native_forward(self,batch):
        for p in self.pools.values():p.counts=tuple(batch['counts'])
        return self.task.native_forward_inputs({'cfp_input':batch['cfp'],'oct_input':batch['oct'],'target':batch['label']})
    def node_identity(self):return [(n.id,n.name) for n in sorted(self.graph.nodes,key=lambda n:n.id)]
    def groups(self, backbone_lr, head_lr, bridge_lr):
        groups=[];seen=set()
        for source in ('cfp','oct','bridge'):
            modules=[(n,m) for n,m in self.task.modules_by_name().items() if n.startswith(source+'_')]
            for is_head in (False,True):
                params=[]
                for name,module in modules:
                    if name.endswith('_logits')!=is_head:continue
                    for p in module.parameters():
                        if p.requires_grad and id(p) not in seen:params.append(p);seen.add(id(p))
                if params:groups.append({'params':params,'lr':bridge_lr if source=='bridge' else head_lr if is_head else backbone_lr,'source':source})
        if seen!={id(p) for p in self.parameters() if p.requires_grad}:raise ValueError('Unowned/duplicate trainable parameters')
        return groups
    def clip(self, maximum):
        for source in ('cfp','oct',*sorted({'_'.join(n.split('_')[:2]) for n in self.task.modules_by_name() if n.startswith('bridge_')})):
            params=[];seen=set()
            for name,module in self.task.modules_by_name().items():
                if name.startswith(source+'_'):
                    for p in module.parameters():
                        if p.requires_grad and id(p) not in seen:params.append(p);seen.add(id(p))
            if params:torch.nn.utils.clip_grad_norm_(params,maximum,error_if_nonfinite=True)
