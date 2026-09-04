"""A small 2D/3D residual network expressed operation-by-operation in MHD V4."""
from __future__ import annotations

import torch
import copy
from torch import nn
from V4.MHD_Framework_V4 import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from .projector import Projector, Return


class Residual(nn.Module):
    def __init__(self, d, cin, cout, stride=1):
        super().__init__()
        conv = nn.Conv2d if d == 2 else nn.Conv3d
        self.main = nn.Sequential(conv(cin, cout, 3, stride, 1, bias=False),
                                  nn.GroupNorm(4, cout), nn.ReLU(),
                                  conv(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(4, cout))
        self.skip = conv(cin, cout, 1, stride, bias=False) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class FlattenEyes(nn.Module):
    def forward(self, x): return x.flatten(0, 1)


class EyePool(nn.Module):
    def forward(self, x):
        return x.flatten(2).mean(-1).reshape(-1, 2, x.shape[1]).mean(1)


class Join(nn.Module):
    def forward(self, a, b): return torch.cat((a, b), 1)


class Add(nn.Module):
    def forward(self, a, b): return a + b


class Mixer(nn.Module):
    def __init__(self, h1, h2, self_only=False):
        super().__init__(); self.h1 = h1
        self.conv = nn.Conv1d(h1 + h2, h1 + h2, 3, padding=1, bias=False)
        nn.init.zeros_(self.conv.weight)
        mask = torch.ones_like(self.conv.weight)
        if self_only:
            mask[:h1, h1:] = 0; mask[h1:, :h1] = 0
        self.register_buffer("mask", mask)

    def forward(self, a, b):
        z = nn.functional.conv1d(torch.cat((a, b), 1), self.conv.weight * self.mask, padding=1)
        return z[:, :self.h1], z[:, self.h1:]


class Loss(nn.Module):
    def forward(self, logits, target):
        return nn.functional.cross_entropy(logits, target.long())


def inflate(module):
    """Inflate ImageNet 2D filters along depth; this is not OCT pretraining."""
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        kd = kh
        result = nn.Conv3d(module.in_channels, module.out_channels, (kd, kh, kw),
                           (module.stride[0], *module.stride),
                           (module.padding[0], *module.padding), bias=module.bias is not None)
        with torch.no_grad():
            result.weight.copy_(module.weight.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd)
            if module.bias is not None: result.bias.copy_(module.bias)
        return result
    if isinstance(module, nn.BatchNorm2d):
        result = nn.BatchNorm3d(module.num_features, eps=module.eps, momentum=module.momentum)
        result.load_state_dict(module.state_dict()); return result
    if isinstance(module, nn.MaxPool2d):
        return nn.MaxPool3d((1, module.kernel_size, module.kernel_size),
                           (1, module.stride, module.stride), (0, module.padding, module.padding))
    result = copy.deepcopy(module)
    for name, child in module.named_children(): setattr(result, name, inflate(child))
    return result


def pretrained_backbones():
    from torchvision.models import resnet18, ResNet18_Weights
    c = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    o = inflate(c)
    # Preserve all B-scans through the stem, then use volumetric layer strides.
    old = o.conv1
    stem = nn.Conv3d(1, 64, old.kernel_size, (1, 2, 2), old.padding, bias=False)
    with torch.no_grad(): stem.weight.copy_(old.weight.sum(1, keepdim=True))
    o.conv1 = stem
    def stages(m):
        return [nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1),
                m.layer2, m.layer3, m.layer4]
    return {"cfp": stages(c), "oct": stages(o)}


class PilotGraph:
    def __init__(self, mode="baseline", seed=3407, device="cpu", backbone="tiny"):
        torch.manual_seed(seed)
        self.nodes, self.edges, self.steps = [], [], []
        self.by_name = {}
        def node(name):
            n = MHD_Node(len(self.nodes), name, MHD_Node.Message(torch.zeros(1)), aggregation="replace")
            self.nodes.append(n); self.by_name[name] = n
            return n.id
        def edge(name, fn, heads, tails):
            eid = len(self.edges)
            self.edges.append(MHD_Edge(eid, name, [MHD_Edge.Operation(fn)]))
            self.steps.append((eid, heads, tails))
        cfp, oct_, target = node("cfp"), node("oct"), node("target")
        features = {}
        # Instantiate both backbones and the common classifier before any
        # bridge parameters so baseline random initialization is identical.
        backbones = {}
        for name, d, cin in [("cfp", 2, 3), ("oct", 3, 1)]:
            blocks = []
            for cout in (8, 16, 32, 64):
                blocks.append(nn.Sequential(Residual(d, cin, cout, 2), Residual(d, cout, cout)))
                cin = cout
            backbones[name] = blocks
        if backbone == "resnet18": backbones = pretrained_backbones()
        bridge_channels = 256 if backbone == "resnet18" else 32
        head = nn.Linear(1024 if backbone == "resnet18" else 128, 2)
        for name, inp in [("cfp", cfp), ("oct", oct_)]:
            out = node(name + "_eye_input"); edge(name + "_flatten_eyes", FlattenEyes(), [inp], [out])
            for stage in range(3):
                nxt = node(f"{name}_stage{stage+1}")
                edge(f"{name}_stage{stage+1}", backbones[name][stage], [out], [nxt]); out = nxt
            features[name] = out
        if mode in ("radon", "scrambled", "self"):
            handoffs, projectors, widths = {}, {}, {}
            shapes = ((6, 6), (8, 6, 6)) if backbone == "resnet18" else ((12, 12), (4, 12, 12))
            for name, shape, mesh in [("cfp", shapes[0], (8,)), ("oct", shapes[1], (4, 4))]:
                p = Projector(shape, mesh, span=16, scramble=mode == "scrambled")
                projectors[name] = p
                width = bridge_channels * p.directions; h = max(1, round(width * .25)); widths[name] = (width, h)
                z = node(name + "_projected"); edge(name + "_project", p, [features[name]], [z])
                u = node(name + "_handoff"); edge(name + "_compress", nn.Conv1d(width, h, 1, bias=False), [z], [u])
                handoffs[name] = u
            c, o = node("cfp_mixed"), node("oct_mixed")
            edge("projection_mixer", Mixer(widths["cfp"][1], widths["oct"][1], mode == "self"),
                 [handoffs["cfp"], handoffs["oct"]], [c, o])
            for name, mixed in [("cfp", c), ("oct", o)]:
                width, h = widths[name]
                expanded = node(name + "_expanded")
                edge(name + "_expand", nn.Conv1d(h, width, 1, bias=False), [mixed], [expanded])
                delta = node(name + "_delta"); edge(name + "_return", Return(projectors[name]), [expanded], [delta])
                updated = node(name + "_updated"); edge(name + "_residual", Add(), [features[name], delta], [updated])
                features[name] = updated
        pooled = []
        for name in ("cfp", "oct"):
            out = node(name + "_stage4")
            edge(name + "_stage4", backbones[name][3], [features[name]], [out])
            pool = node(name + "_participant"); edge(name + "_pool", EyePool(), [out], [pool]); pooled.append(pool)
        joined = node("joined"); edge("join", Join(), pooled, [joined])
        logits = node("logits"); edge("classifier", head, [joined], [logits])
        loss = node("loss"); edge("criterion", Loss(), [logits, target], [loss])
        roles, sorts = [], []
        for eid, heads, tails in self.steps:
            r = torch.zeros((len(self.edges), len(self.nodes)), dtype=torch.long)
            s = torch.zeros_like(r)
            for order, nid in enumerate(heads): r[eid, nid] = -1; s[eid, nid] = order
            for order, nid in enumerate(tails, len(heads)): r[eid, nid] = 1; s[eid, nid] = order
            roles.append(r); sorts.append(s)
        self.forward_levels = list(range(len(roles)))
        self.backward_levels = list(range(len(roles), 2 * len(roles)))
        self.graph = MHD_Graph(set(self.nodes), set(self.edges),
                              {MHD_Topo(roles + [-r for r in reversed(roles)], sorts + list(reversed(sorts)))},
                              device=torch.device(device))

    def set_inputs(self, cfp, oct_, target):
        for name, x in zip(("cfp", "oct", "target"), (cfp, oct_, target)):
            n = self.by_name[name]
            n.feature_message.initial_state = x
            n.feature_message.current_state = x

    def forward(self, cfp, oct_, target):
        self.set_inputs(cfp, oct_, target)
        self.graph.forward(levels=self.forward_levels)
        return self.by_name["logits"].feature_message.current_state, self.by_name["loss"].feature_message.current_state

    def backward(self): self.graph.backward(levels=self.backward_levels)

    def modules_by_name(self):
        return {e.name: e.edge_operations[0].function for e in self.edges}

    def save_state(self):
        return {k: {n: v.detach().cpu().clone() for n,v in m.state_dict().items()}
                for k,m in self.modules_by_name().items()}

    def load_state(self, state):
        for k,m in self.modules_by_name().items():
            if k in state: m.load_state_dict(state[k], strict=True)

    def native_forward(self, cfp, oct_, target):
        values = {self.by_name[k].id:v for k,v in zip(("cfp","oct","target"),(cfp,oct_,target))}
        for eid, heads, tails in self.steps:
            fn = self.edges[eid].edge_operations[0].function
            out = fn(*[values[n] for n in heads])
            for n, x in zip(tails, out if isinstance(out, tuple) else (out,)): values[n] = x
        return values[self.by_name["logits"].id], values[self.by_name["loss"].id]
