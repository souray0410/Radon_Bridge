"""Explicit task-local optimization policies; no hidden cross-task clipping."""
import torch

def configure_optimizer(g, protocol, warm=False):
    stages = protocol['warmup_adapt_stages' if warm else 'adapt_stages']
    if not stages or any(s not in (1, 2, 3, 4) for s in stages):
        raise ValueError('Explicit trainable backbone stages required')
    groups = []
    for name, module in g.modules_by_name().items():
        backbone = any(name == f'{k}_stage{s}' for k in g.branches for s in stages)
        enabled = backbone or name.endswith('_head') or name.startswith('bridge_')
        params = list(module.parameters())
        for p in params:
            p.requires_grad_(enabled)
        if enabled and params:
            groups.append({'params': params, 'lr': protocol['backbone_lr'] if backbone else protocol['head_bridge_lr']})
    return torch.optim.AdamW(groups, weight_decay=protocol.get('weight_decay', .01))

def clip_task_gradients(g, max_norm=5.):
    if g.head_mode != 'separate':
        raise ValueError('Per-task clipping requires separate task paths')
    groups = {k: [] for k in g.branches}
    groups['communication'] = []
    seen = set()
    for name, module in g.modules_by_name().items():
        key = 'communication' if name.startswith('bridge_') else next((k for k in g.branches if name.startswith(k + '_')), None)
        if key is None:
            continue
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                groups[key].append(p); seen.add(id(p))
    required = {id(p) for p in g.graph.parameters() if p.requires_grad}
    if required != seen:
        raise ValueError('Unassigned trainable parameter')
    return {k: torch.nn.utils.clip_grad_norm_(v, max_norm, error_if_nonfinite=True)
            for k, v in groups.items() if v}
