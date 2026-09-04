"""MHD public backward must accumulate sample-weighted microbatch gradients."""
import json
import torch
from torch import nn
from radonbridge.graph import MHDBuilder
from radonbridge.model import TaskLoss

torch.manual_seed(314)
b=MHDBuilder();x=b.node('x');a=b.node('a');loss=b.node('loss')
b.edge('linear',nn.Linear(3,2,bias=False),[x],[a])
class Criterion(nn.Module):
    def forward(self,x):return x.square().mean()
y=b.node('criterion');b.edge('criterion',Criterion(),[a],[y]);objective=TaskLoss();b.edge('loss',objective,[y],[loss])
g=b.compile('cpu').double();inputs=torch.randn(7,3,dtype=torch.float64)
objective.scale=1.;b.set_inputs({'x':inputs});g.forward(levels=b.forward_levels);g.backward(levels=b.backward_levels)
reference=[p.grad.clone() for p in g.parameters()];g.zero_grad(set_to_none=True)
for part in [inputs[:3],inputs[3:6],inputs[6:]]:
    objective.scale=len(part)/len(inputs);b.set_inputs({'x':part});g.forward(levels=b.forward_levels);g.backward(levels=b.backward_levels)
error=max(float((p.grad-r).abs().max()) for p,r in zip(g.parameters(),reference));assert error<1e-12
print(json.dumps({'sample_weighted_mhd_accumulation':True,'maximum_gradient_error':error}))
