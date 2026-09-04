"""A small 2D/3D residual network expressed operation-by-operation in MHD V4."""
from __future__ import annotations

import torch
import copy
from torch import nn
from V4.MHD_Framework_V4 import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from .projector import Projector, Return
from .bridge import FeatureSpec, attach_group
from .graph import MHDBuilder


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


class MeanLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", "sum"): raise ValueError(reduction)
        self.reduction = reduction
    def forward(self, *losses):
        return torch.stack(losses).sum() if self.reduction == "sum" else torch.stack(losses).mean()


class PilotGraph:
    def __init__(self, mode="baseline", seed=3407, device="cpu", backbone="tiny", handoff_ratio=.25, cfp_size=96, modalities="both", head_mode="separate", bridge_stages=(3,), upsilon=None, mesh_references=None, loss_reduction="mean", mixer_kernel_size=3):
        if modalities not in ("both", "cfp", "oct"): raise ValueError(modalities)
        if cfp_size not in (96, 224): raise ValueError(cfp_size)
        if modalities != "both" and mode != "baseline": raise ValueError("Bridge requires both modalities")
        if head_mode not in ("separate", "shared_legacy"): raise ValueError(head_mode)
        if tuple(sorted(set(bridge_stages))) != tuple(bridge_stages) or any(s not in (1,2,3) for s in bridge_stages):
            raise ValueError("Bridge locations must be ordered unique intermediate stages 1..3")
        if head_mode == "shared_legacy" and tuple(bridge_stages) != (3,): raise ValueError("Legacy layout only supports stage3")
        self.head_mode=head_mode
        self.branches=branches = ("cfp", "oct") if modalities == "both" else (modalities,)
        torch.manual_seed(seed)
        builder=MHDBuilder()
        self.nodes,self.edges,self.steps,self.by_name=builder.nodes,builder.edges,builder.steps,builder.by_name
        node,edge=builder.node,builder.edge
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
        last_channels=512 if backbone == "resnet18" else 64
        # Build all task heads before any communication parameters. Their initial
        # states are identical whether branches are trained alone or together.
        if head_mode == "separate": heads={name:nn.Linear(last_channels,2) for name in ("cfp","oct")}
        else: head=nn.Linear(last_channels*len(branches),2)
        for name,inp in [("cfp",cfp),("oct",oct_)]:
            if name not in branches:continue
            out=node(name+"_eye_input");edge(name+"_flatten_eyes",FlattenEyes(),[inp],[out]);features[name]=out
        self.communication_groups=[]
        def bridge(stage):
            prefix="" if head_mode == "shared_legacy" else f"bridge_s{stage}_"
            channels=([64,128,256][stage-1] if backbone=="resnet18" else [8,16,32][stage-1])
            if backbone=="resnet18":
                stride=2**(stage+1);depth=32//(2**(stage-1))
            else:stride=2**stage;depth=32//stride
            shapes=((cfp_size//stride,)*2,(depth,96//stride,96//stride))
            if head_mode == "separate":
                references=mesh_references or {"cfp":(8,),"oct":(4,4)}
                specs=[FeatureSpec(name,channels,shape,tuple(references[name])) for name,shape in zip(("cfp","oct"),shapes)]
                outputs,metadata=attach_group(node,edge,specs,features,prefix,upsilon or (1.,1.,handoff_ratio),mode,kernel_size=mixer_kernel_size)
                features.update(outputs);self.communication_groups.append(metadata|{"stage":stage})
                return
            handoffs,projectors,widths={},{},{}
            for name,shape,mesh in [("cfp",shapes[0],(8,)),("oct",shapes[1],(4,4))]:
                projector=Projector(shape,mesh,span=16,scramble=mode=="scrambled")
                projectors[name]=projector
                width=channels*projector.directions;h=max(1,round(width*handoff_ratio));widths[name]=(width,h)
                z=node(prefix+name+"_projected");edge(prefix+name+"_project",projector,[features[name]],[z])
                u=node(prefix+name+"_handoff");edge(prefix+name+"_compress",nn.Conv1d(width,h,1,bias=False),[z],[u]);handoffs[name]=u
            c,o=node(prefix+"cfp_mixed"),node(prefix+"oct_mixed")
            edge(prefix+"projection_mixer",Mixer(widths["cfp"][1],widths["oct"][1],mode=="self"),[handoffs["cfp"],handoffs["oct"]],[c,o])
            for name,mixed in [("cfp",c),("oct",o)]:
                width,h=widths[name]
                expanded=node(prefix+name+"_expanded");edge(prefix+name+"_expand",nn.Conv1d(h,width,1,bias=False),[mixed],[expanded])
                delta=node(prefix+name+"_delta");edge(prefix+name+"_return",Return(projectors[name]),[expanded],[delta])
                updated=node(prefix+name+"_updated");edge(prefix+name+"_residual",Add(),[features[name],delta],[updated]);features[name]=updated
        for stage in range(1,5):
            for name in branches:
                out=node(f"{name}_stage{stage}");edge(f"{name}_stage{stage}",backbones[name][stage-1],[features[name]],[out]);features[name]=out
            if stage in bridge_stages and mode in ("radon","scrambled","self"):bridge(stage)
        pooled=[];losses=[]
        for name in branches:
            pool=node(name+"_participant");edge(name+"_pool",EyePool(),[features[name]],[pool]);pooled.append(pool)
            if head_mode == "separate":
                logits=node(name+"_logits");edge(name+"_head",heads[name],[pool],[logits])
                task_loss=node(name+"_loss");edge(name+"_criterion",Loss(),[logits,target],[task_loss]);losses.append(task_loss)
        loss=node("loss")
        if head_mode == "separate":
            edge("task_loss_"+loss_reduction,MeanLoss(loss_reduction),losses,[loss])
        else:
            joined=node("joined");edge("join",Join() if len(branches)==2 else nn.Identity(),pooled,[joined])
            logits=node("logits");edge("classifier",head,[joined],[logits]);edge("criterion",Loss(),[logits,target],[loss])
        self.graph=builder.compile(device)
        self.forward_levels=builder.forward_levels;self.backward_levels=builder.backward_levels

    def set_inputs(self, cfp, oct_, target):
        for name, x in zip(("cfp", "oct", "target"), (cfp, oct_, target)):
            n = self.by_name[name]
            n.feature_message.initial_state = x
            n.feature_message.current_state = x

    def forward(self, cfp, oct_, target):
        self.set_inputs(cfp, oct_, target)
        self.graph.forward(levels=self.forward_levels)
        logits=({k:self.by_name[k+"_logits"].feature_message.current_state for k in self.branches}
                if self.head_mode=="separate" else self.by_name["logits"].feature_message.current_state)
        return logits,self.by_name["loss"].feature_message.current_state

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
        logits=({k:values[self.by_name[k+"_logits"].id] for k in self.branches}
                if self.head_mode=="separate" else values[self.by_name["logits"].id])
        return logits,values[self.by_name["loss"].id]
