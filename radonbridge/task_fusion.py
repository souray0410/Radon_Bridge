"""Task-level fusion readouts; distinct from in-place communication adapters.

Gated pooling follows Ilse et al. (ICML 2018), eq. 9. Instances here are
native stage4 spatial feature cells, grouped by participant including both eyes.
This is not a reproduction of MM-MIL's 2D B-scan/crop encoders.
"""
import torch
from torch import nn


def participant_tokens(x):
    if x.shape[0] % 2:
        raise ValueError('Expected two ordered eyes per participant')
    return x.flatten(2).transpose(1, 2).reshape(x.shape[0] // 2, -1, x.shape[1])


class GatedPool(nn.Module):
    def __init__(self, channels, attention_dimension):
        super().__init__()
        self.v = nn.Linear(channels, attention_dimension)
        self.u = nn.Linear(channels, attention_dimension)
        self.w = nn.Linear(attention_dimension, 1, bias=False)

    def forward(self, tokens):
        scores = self.w(torch.tanh(self.v(tokens)) * torch.sigmoid(self.u(tokens)))
        weights = scores.softmax(dim=1)
        return (weights * tokens).sum(dim=1)


class TaskFusionHead(nn.Module):
    def __init__(self, *, pooling, hidden_dimension=256, attention_dimension=128,
                 seed=0, channels=(512, 512)):
        super().__init__()
        if pooling not in ('mean', 'gated_mil'):
            raise ValueError('Unknown task fusion pooling')
        if hidden_dimension <= 0 or attention_dimension <= 0:
            raise ValueError('Fusion dimensions must be positive')
        self.pooling = pooling
        # Local RNG scopes preserve bridge construction and training random streams.
        # Identical classifier initialization across all methods at a given seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 170001)
            self.classifier = nn.Sequential(nn.Linear(sum(channels), hidden_dimension),
                                            nn.ReLU(), nn.Linear(hidden_dimension, 2))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 170003)
            self.pools = nn.ModuleList([GatedPool(c, attention_dimension) for c in channels]) if pooling == 'gated_mil' else nn.ModuleList()

    def forward(self, *features):
        tokens = [participant_tokens(x) for x in features]
        pooled = [x.mean(1) for x in tokens] if self.pooling == 'mean' else [p(x) for p, x in zip(self.pools, tokens)]
        return self.classifier(torch.cat(pooled, dim=1))
