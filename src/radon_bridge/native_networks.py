"""Current CFP/OCT native network provider; contains no communication attachment."""
import torch
from torch import nn
from .backbones import pretrained_backbones, FlattenEyes, EyePool, Loss
from .graph import MHDBuilder
from .network_interface import NativeNetwork

class TaskLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.scale=1.
    def forward(self,*losses):return torch.stack(losses).sum()*self.scale


def build_cfp_oct(*,seed=3407,backbone='resnet18',task_fusion=None):
    # Preserve legacy construction order and random initialization exactly.
    torch.manual_seed(seed)
    branches=('cfp','oct')
    builder = MHDBuilder()
    node, edge = builder.node, builder.edge
    inputs = {k: node(k) for k in (*branches, 'target')}
    backbones = pretrained_backbones(backbone)
    heads = {k: nn.Linear(512, 2) for k in branches}
    features = {}
    for k in branches:
        features[k] = node(k+'_eye_input')
        edge(k+'_flatten_eyes', FlattenEyes(), [inputs[k]], [features[k]])
    for stage in range(1, 5):
        for k in branches:
            out = node(f'{k}_stage{stage}')
            edge(f'{k}_stage{stage}', backbones[k][stage-1], [features[k]], [out])
            features[k] = out
    losses = []
    for k in branches:
        pool = node(k+'_participant'); edge(k+'_pool', EyePool(), [features[k]], [pool])
        logits = node(k+'_logits'); edge(k+'_head', heads[k], [pool], [logits])
        loss = node(k+'_loss'); edge(k+'_criterion', Loss(), [logits, inputs['target']], [loss]); losses.append(loss)
    if task_fusion is not None:
        from .task_fusion import TaskFusionHead
        if set(task_fusion) != {'pooling', 'hidden_dimension', 'attention_dimension'}:
            raise ValueError('Explicit task fusion pooling and dimensions required')
        fused = node('fusion_logits')
        edge('fusion_head', TaskFusionHead(seed=seed, **task_fusion),
             [features[k] for k in branches], [fused])
        fused_loss = node('fusion_loss')
        edge('fusion_criterion', Loss(), [fused, inputs['target']], [fused_loss])
        losses.append(fused_loss)
    loss = node('loss'); edge('task_loss_sum', TaskLoss(), losses, [loss])

    outputs=(*branches,'fusion') if task_fusion is not None else branches
    native_names=tuple(e.name for e in builder.edges if not e.name.startswith('fusion_'))
    return NativeNetwork(builder=builder,input_names=('cfp','oct','target'),
        prediction_nodes={k:k+'_logits' for k in outputs},loss_node='loss',loss_scaler_edge='task_loss_sum',
        branches=branches,source_modules={k:tuple(n for n in native_names if n.startswith(k+'_')) for k in branches},
        native_checkpoint_modules=native_names,
        probe_inputs={'cfp':torch.zeros(1,2,3,224,224),'oct':torch.zeros(1,2,1,32,96,96),'target':torch.zeros(1,dtype=torch.long)},
        metadata={'provider':'cfp_oct_resnet18_v1','task_fusion':task_fusion,'head_mode':'separate'})
