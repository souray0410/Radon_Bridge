"""Native 2D/3D operations shared by the current MHD network provider."""
import copy
import torch
from torch import nn

class FlattenEyes(nn.Module):
    def forward(self, x): return x.flatten(0, 1)

class EyePool(nn.Module):
    def forward(self, x):
        return x.flatten(2).mean(-1).reshape(-1, 2, x.shape[1]).mean(1)

class Loss(nn.Module):
    def forward(self, logits, target):
        return nn.functional.cross_entropy(logits, target.long())

class SliceMaxPool3d(nn.Module):
    """Depth-one 3D max pooling via deterministic per-slice 2D kernels."""
    def __init__(self, pool):
        super().__init__()
        if pool.return_indices:
            raise ValueError('Index-returning pooling requires explicit 3D index conversion')
        self.pool = copy.deepcopy(pool)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = self.pool(x.permute(0, 2, 1, 3, 4).reshape(b*d, c, h, w))
        return y.reshape(b, d, c, *y.shape[-2:]).permute(0, 2, 1, 3, 4).contiguous()

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
        return SliceMaxPool3d(module)
    result = copy.deepcopy(module)
    for name, child in module.named_children(): setattr(result, name, inflate(child))
    return result

def pretrained_backbones(name="resnet18"):
    from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
    if name == "resnet18": c = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet34": c = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    else: raise ValueError(name)
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
