"""Complete independent task networks with configurable in-place bridge groups."""
import torch
from torch import nn
from .legacy_model import pretrained_backbones, FlattenEyes, EyePool, Loss, MeanLoss
from .bridge import attach_to_nodes
from .graph import MHDBuilder


class TaskLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = 1.
    def forward(self, *losses):
        return torch.stack(losses).sum()*self.scale


class PilotGraph:
    def __init__(self, *, bridge_configs=(), seed=3407, device='cpu', backbone='resnet18'):
        self.head_mode, self.branches = 'separate', ('cfp', 'oct')
        torch.manual_seed(seed)
        builder = self.builder = MHDBuilder()
        self.nodes, self.edges, self.steps, self.by_name = builder.nodes, builder.edges, builder.steps, builder.by_name
        node, edge = builder.node, builder.edge
        inputs = {k: node(k) for k in (*self.branches, 'target')}
        backbones = pretrained_backbones(backbone)
        heads = {k: nn.Linear(512, 2) for k in self.branches}
        features = {}
        for k in self.branches:
            features[k] = node(k+'_eye_input')
            edge(k+'_flatten_eyes', FlattenEyes(), [inputs[k]], [features[k]])
        for stage in range(1, 5):
            for k in self.branches:
                out = node(f'{k}_stage{stage}')
                edge(f'{k}_stage{stage}', backbones[k][stage-1], [features[k]], [out])
                features[k] = out
        losses = []
        for k in self.branches:
            pool = node(k+'_participant'); edge(k+'_pool', EyePool(), [features[k]], [pool])
            logits = node(k+'_logits'); edge(k+'_head', heads[k], [pool], [logits])
            loss = node(k+'_loss'); edge(k+'_criterion', Loss(), [logits, inputs['target']], [loss]); losses.append(loss)
        loss = node('loss'); edge('task_loss_sum', TaskLoss(), losses, [loss])
        self.communication_groups = []
        configs = list(bridge_configs)
        if configs:
            modules = list(self.modules_by_name().values())
            training = {m: m.training for module in modules for m in module.modules()}
            for m in training: m.training = False
            try:
                with torch.no_grad():
                    samples = builder.native_forward({'cfp': torch.zeros(1,2,3,224,224),
                        'oct': torch.zeros(1,2,1,32,96,96), 'target': torch.zeros(1,dtype=torch.long)})
            finally:
                for m, value in training.items(): m.training = value
            for index, cfg in enumerate(configs):
                if set(cfg) != {'nodes', 'M', 'S', 'rho', 'mode'}:
                    raise ValueError('Each bridge requires exactly nodes, M, S, rho, mode')
                _, metadata = attach_to_nodes(builder, cfg['nodes'], prefix=f'bridge_{index}_', samples=samples,
                                               M=cfg['M'], S=cfg['S'], rho=cfg['rho'], mode=cfg['mode'])
                self.communication_groups.append(metadata)
            del samples
        self.graph = builder.compile(device).float()
        self.forward_levels, self.backward_levels = builder.forward_levels, builder.backward_levels

    def set_inputs(self, cfp, oct_, target):
        self.builder.set_inputs(dict(zip(('cfp','oct','target'), (cfp,oct_,target))))

    def forward(self, cfp, oct_, target, loss_scale=1.):
        self.modules_by_name()['task_loss_sum'].scale = loss_scale
        self.set_inputs(cfp,oct_,target)
        self.graph.forward(levels=self.forward_levels)
        return {k:self.by_name[k+'_logits'].feature_message.current_state for k in self.branches}, self.by_name['loss'].feature_message.current_state

    def backward(self):
        self.graph.backward(levels=self.backward_levels)

    def modules_by_name(self):
        return {e.name:e.edge_operations[0].function for e in self.edges}

    def save_state(self):
        return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in self.modules_by_name().items()}

    def load_native_state(self, state, branch=None):
        modules={k:m for k,m in self.modules_by_name().items() if not k.startswith('bridge_')}
        if branch is not None:
            modules={k:m for k,m in modules.items() if k.startswith(branch+'_')}
        if set(state) != set(modules):
            raise ValueError(f'Checkpoint native module mismatch: missing={set(modules)-set(state)}, extra={set(state)-set(modules)}')
        for key,module in modules.items():module.load_state_dict(state[key],strict=True)

    def native_forward(self, cfp, oct_, target):
        values = self.builder.native_forward(dict(zip(('cfp','oct','target'), (cfp,oct_,target))))
        return {k:values[k+'_logits'] for k in self.branches}, values['loss']
