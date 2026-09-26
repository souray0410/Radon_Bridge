"""Complete two-branch MMTM classification host on native CFP2D/OCT3D graphs.

This is a UKB task adaptation, not reproduction of the author's action dataset.
The historical single-site, identity-initialized MMTM remains a distinct method.
"""
import torch
from torch import nn

from radon_bridge.models.native_pair import NativePair, definition
from radon_bridge.models.task import MHDTaskGraph


class MeanLogits(nn.Module):
    def forward(self, first, second):
        if first.shape != second.shape:
            raise ValueError('Matched participant logits required')
        return (first + second) / 2


class ScaledCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.

    def forward(self, logits, labels):
        return nn.functional.cross_entropy(logits, labels) * self.scale


class ModernMMTMHost(NativePair):
    def __init__(self, parents, shapes, *, sites=('stage2', 'stage3', 'stage4'),
                 reduction_ratio=4, frozen=False, device='cpu', additions=None):
        nn.Module.__init__(self)
        if not sites or len(set(sites)) != len(sites):
            raise ValueError('Explicit distinct corresponding sites required')
        native, self.pools = definition(parents, shapes)
        builder = native.builder
        for site in sites:
            if any(role + '_' + site not in builder.by_name for role in ('cfp', 'oct')):
                raise ValueError('Corresponding endpoint missing: ' + site)
        # Replace two separate-task losses with one classification loss of the
        # averaged logits, matching the author-code default decision function.
        remove = {'cfp_criterion', 'oct_criterion', 'task_loss_sum'}
        removed = {e.id for e in builder.edges if e.name in remove}
        old_edges = [e for e in builder.edges if e.id not in removed]
        remap = {e.id: i for i, e in enumerate(old_edges)}
        builder.steps = [(remap[eid], heads, tails) for eid, heads, tails in builder.steps
                         if eid not in removed]
        builder.level_groups = {remap[eid]: group for eid, group in builder.level_groups.items()
                                if eid in remap}
        for edge in old_edges:
            edge.id = remap[edge.id]
        builder.edges = old_edges
        fused = builder.node('joint_logits')
        loss = builder.node('joint_loss')
        builder.edge('joint_mean_logits', MeanLogits(),
                     [builder.by_name[r + '_logits'].id for r in ('cfp', 'oct')], [fused])
        builder.edge('joint_criterion', ScaledCrossEntropy(),
                     [fused, builder.by_name['target'].id], [loss])
        native.prediction_nodes = {'joint': 'joint_logits', 'cfp': 'cfp_logits', 'oct': 'oct_logits'}
        native.loss_node = 'joint_loss'
        native.loss_scaler_edge = 'joint_criterion'
        native.native_checkpoint_modules = tuple(e.name for e in builder.edges)
        communications = [dict(family='mmtm_author', nodes=[r + '_' + site for r in ('cfp', 'oct')],
                               reduction_ratio=reduction_ratio) for site in sites]
        additions = {} if additions is None else additions
        if set(additions) - set(sites):
            raise ValueError('Addition must identify a corresponding host site')
        self.addition_prefixes = []
        for index, site in enumerate(sites):
            if site not in additions:
                continue
            config = dict(additions[site])
            if {'nodes', 'parallel_to', 'family'} & set(config):
                raise ValueError('Addition topology is owned by the matched host')
            self.addition_prefixes.append(f'bridge_{len(communications)}_')
            communications.append(dict(config, family='radon', parallel_to=index,
                                       nodes=[r+'_'+site for r in ('cfp','oct')]))
        self.task = MHDTaskGraph(native, bridge_configs=communications, device=device)
        self.graph = self.task.graph
        self.parent_node_map = native.metadata['parent_node_map']
        self.sites = tuple(sites)
        self.selection_output = "joint"
        self.frozen = frozen
        if frozen:
            for name, module in self.task.modules_by_name().items():
                if not name.startswith('bridge_'):
                    module.requires_grad_(False)
        self.train(False)

    def _inputs(self, batch):
        counts = tuple(batch['counts'])
        if len(counts) != len(batch['label']) or sum(counts) != len(batch['cfp']) or len(batch['cfp']) != len(batch['oct']):
            raise ValueError('Paired observed eyes and participants required')
        for pool in self.pools.values():
            pool.counts = counts
        return {'cfp_input': batch['cfp'], 'oct_input': batch['oct'], 'target': batch['label']}

    def forward(self, batch, loss_scale=1.):
        return self.task.forward_inputs(self._inputs(batch), loss_scale)

    def native_forward(self, batch, loss_scale=1.):
        self.task.modules_by_name()['joint_criterion'].scale = loss_scale
        return self.task.native_forward_inputs(self._inputs(batch))

    def backward(self):
        self.task.backward()

    def load_matched_host(self, state):
        """Strict one-time transfer from selected A; new deltas remain initialized.

        This is an explicit scientific branch operation, not a checkpoint-resume
        fallback. Complete augmented executions use normal strict state loading.
        """
        modules = self.task.modules_by_name()
        expected = {name for name in modules
                    if not any(name.startswith(prefix) for prefix in self.addition_prefixes)}
        if set(state) != expected:
            raise ValueError('Selected host modules differ from matched A')
        # Validate the complete structure before any parameter is updated.
        for name in expected:
            current = modules[name].state_dict()
            if set(state[name]) != set(current):
                raise ValueError('Selected host state keys changed: '+name)
            for key, tensor in current.items():
                if state[name][key].shape != tensor.shape or state[name][key].dtype != tensor.dtype:
                    raise ValueError('Selected host tensor changed: '+name+'.'+key)
        for name in expected:
            modules[name].load_state_dict(state[name], strict=True)
