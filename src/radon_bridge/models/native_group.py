"""Compose independent complete task graphs, preserving role-specific labels."""
import copy
import re
import torch
from torch import nn
from radon_bridge.models.graph import MHDBuilder
from radon_bridge.models.interface import NativeNetwork
from radon_bridge.models.task import MHDTaskGraph
from radon_bridge.models.native_pair import NativePair, ParticipantPool
from radon_bridge.models.networks import TaskLoss


class GroupLoss(TaskLoss):
    weights = None
    def forward(self, *losses):
        weights = (1.,)*len(losses) if self.weights is None else self.weights
        if len(weights) != len(losses): raise ValueError('Task loss weights do not match sources')
        return sum(w*loss for w,loss in zip(weights,losses)) * self.scale


def stage_aliases(original):
    result = {}
    for stage in (2, 3, 4):
        names = [f'stage{stage}', f'denseblock{stage}']
        found = [n for n in names if n in original.endpoint_nodes]
        if len(found) != 1:
            raise ValueError(f'Unverified stage mapping {original.config.name}: {names}')
        result[f'stage{stage}'] = original.endpoint_nodes[found[0]]
    return result


def definition(parents, shapes, sources):
    keys = [s['key'] for s in sources]
    if len(keys) < 2 or len(set(keys)) != len(keys) or set(parents) != set(keys) or set(shapes) != set(keys):
        raise ValueError('Explicit unique sources/parents/shapes required')
    if any(not re.fullmatch(r'[a-z][a-z0-9_]*', k) or k.startswith('bridge_') for k in keys):
        raise ValueError('Invalid or reserved source key')
    builder = MHDBuilder(); pools = {}; ownership = {}; mapping = {}; predictions = {}; losses = []
    inputs = {}; head_edges = {}; channel_axes = {}
    for src in sources:
        key = src['key']; original = parents[key].graph
        if original.config.views != 1 or original.config.spatial_dims != src['spatial_dims']:
            raise ValueError('Parent dimension/view contract mismatch')
        local = {old.id: builder.node(key+'_'+old.name) for old in sorted(original.nodes, key=lambda n: n.id)}
        mapping[key] = {str(old.id): local[old.id] for old in original.nodes}
        aliases = dict(original.endpoint_nodes, **stage_aliases(original))
        for alias, old in aliases.items():
            builder.by_name[key+'_'+alias] = builder.nodes[local[old]]
        for stage in ('stage2', 'stage3', 'stage4'):
            channel_axes[key+'_'+stage] = -1 if src['model'].startswith('swin_') and src['spatial_dims'] == 2 else 1
        feature = original.endpoint_nodes['features']
        pooled = builder.node(key+'_participant'); pool = ParticipantPool(); pools[key] = pool
        ownership[key] = []; head_edges[key] = []
        by_id = {e.id: e for e in original.edges}
        for eid, heads, tail in original._definitions:
            edge = by_id[eid]; name = key+'_'+edge.name
            builder.edge(name, copy.deepcopy(edge.edge_operations[0].function),
                         [pooled if h == feature else local[h] for h in heads], [local[tail]])
            ownership[key].append(name)
            if tail == original.endpoint_nodes['logits']:
                head_edges[key].append(name)
            if tail == feature:
                name = key+'_participant_pool'
                builder.edge(name, pool, [local[feature]], [pooled]); ownership[key].append(name)
        target = builder.node(key+'_target'); loss = builder.node(key+'_loss')
        builder.edge(key+'_criterion', nn.CrossEntropyLoss(), [local[original.endpoint_nodes['logits']], target], [loss])
        losses.append(loss); predictions[key] = key+'_logits'
        inputs[key+'_input'] = torch.zeros(1, *shapes[key])
        inputs[key+'_target'] = torch.zeros(1, dtype=torch.long)
    loss = builder.node('loss'); builder.edge('task_loss_sum', GroupLoss(), losses, [loss])
    native = NativeNetwork(builder, tuple(inputs), predictions, 'loss', inputs, tuple(keys),
        tuple(e.name for e in builder.edges), {k: tuple(v) for k, v in ownership.items()},
        'task_loss_sum', dict(head_mode='separate', parent_node_map=mapping, sources=copy.deepcopy(sources),
                             head_edges=head_edges, channel_axes=channel_axes))
    return native, pools


class NativeGroup(NativePair):
    def to(self, *args, **kwargs):
        if hasattr(self, 'device_placement'):
            raise ValueError('Placed source graphs must be rebuilt to change device topology')
        self.graph.to(*args, **kwargs)
        return self

    def cpu(self):
        return self.to('cpu')

    def __init__(self, parents, shapes, sources, bridges=(), *, device='cpu', frozen=False):
        nn.Module.__init__(self)
        native, self.pools = definition(parents, shapes, sources)
        self.sources = copy.deepcopy(sources); self.source_keys = tuple(s['key'] for s in sources)
        self.task = MHDTaskGraph(native, bridge_configs=bridges, device=device)
        self.graph = self.task.graph; self.parent_node_map = native.metadata['parent_node_map']
        self.frozen = frozen
        if frozen:
            for name, module in self.task.modules_by_name().items():
                if not name.startswith('bridge_'):
                    for p in module.parameters(): p.requires_grad_(False)
        self.train(False)

    def inputs(self, batch):
        if set(batch['inputs']) != set(self.source_keys) or set(batch['labels']) != set(self.source_keys):
            raise ValueError('All sources require independent inputs and targets')
        values = {}
        for key in self.source_keys:
            self.pools[key].counts = tuple(batch['counts'])
            values[key+'_input'] = batch['inputs'][key]
            values[key+'_target'] = batch['labels'][key]
            if hasattr(self, 'device_placement'):
                device = self.device_placement['sources'][key]
                from radon_bridge.models.placement import transfer
                values[key+'_input'] = transfer(values[key+'_input'], device, self.device_placement['transport'])
                values[key+'_target'] = transfer(values[key+'_target'], device, self.device_placement['transport'])
        return values

    def forward(self, batch, loss_scale=1.):
        return self.task.forward_inputs(self.inputs(batch), loss_scale)

    def native_forward(self, batch):
        return self.task.native_forward_inputs(self.inputs(batch))

    def groups(self, backbone_lr, head_lr, bridge_lr):
        groups = []; seen = set(); modules = self.task.modules_by_name()
        for key in self.source_keys:
            head = set(self.task.definition.metadata['head_edges'][key])
            for is_head in (False, True):
                params = []
                for name in self.task.definition.source_modules[key]:
                    if (name in head) != is_head: continue
                    for p in modules[name].parameters():
                        if p.requires_grad and id(p) not in seen: params.append(p); seen.add(id(p))
                if params: groups.append(dict(params=params, lr=head_lr if is_head else backbone_lr, source=key))
        for prefix in sorted({n.split('_exchange')[0] for n in modules if n.startswith('bridge_') and n.endswith('_exchange')}):
            params = [p for p in modules[prefix+'_exchange'].parameters() if p.requires_grad and id(p) not in seen]
            seen.update(map(id, params))
            if params: groups.append(dict(params=params, lr=bridge_lr, source=prefix))
        if seen != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError('Unowned trainable parameter')
        return groups

    def clip(self, maximum):
        combined = {}
        for group in self.groups(1., 1., 1.): combined.setdefault(group['source'], []).extend(group['params'])
        for params in combined.values(): torch.nn.utils.clip_grad_norm_(params, maximum, error_if_nonfinite=True)
