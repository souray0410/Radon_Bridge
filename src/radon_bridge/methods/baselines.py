"""Single-stage communication adapters, preserving native task predictions.

MMTM follows the paper's 2*sigmoid excitation, with identity initialization.
Attention is a simultaneous bidirectional pre-LN attention adapter, not a
complete Transformer. Neither adapter claims width matching with Radon.
"""
import math
import torch
from torch import nn
from radon_bridge.methods.projection import positive_integer


class NativeExchange(nn.Module):
    def __init__(self,specs,family):
        super().__init__()
        self.keys=[s.key for s in specs]
        if len(specs)!=2 or len(set(self.keys))!=2:raise ValueError('This baseline requires two distinct sources')
        self.shapes=[(s.channels,*s.shape) for s in specs]
        self.lengths=[math.prod(s) for s in self.shapes]
        self.family=family;self.compression=None
        self.latest_inputs=self.latest_deltas=None
        self.metadata={'family':family,'compression':None,'M':None,'S':None,'rho':None,
            'participants':[{'key':s.key,'channels':s.channels,'shape':list(s.shape),'tokens':math.prod(s.shape)} for s in specs]}

    def finish_metadata(self):
        n=sum(p.numel() for p in self.parameters())
        self.metadata.update(stored_bridge_parameters=n,effective_bridge_parameters=n)

    def export_fixed_bases(self,directory):return []

    def check(self,features):
        if len(features)!=len(self.shapes) or any(tuple(x.shape[1:])!=s for x,s in zip(features,self.shapes)):
            raise ValueError('Participant shapes changed')
        if len({x.shape[0] for x in features})!=1:raise ValueError('Participants must share batch alignment')

    def packet(self,features,deltas):
        self.latest_inputs=tuple(x.detach() for x in features)
        self.latest_deltas=tuple(x.detach() for x in deltas)
        return torch.cat([(x+d).flatten(1) for x,d in zip(features,deltas)],1)


class MMTMExchange(NativeExchange):
    def __init__(self,specs,reduction_ratio):
        super().__init__(specs,'mmtm');positive_integer(reduction_ratio,'reduction_ratio',1)
        hidden=int(2*sum(s.channels for s in specs)/reduction_ratio)
        if hidden<1:raise ValueError('MMTM hidden dimension must be positive')
        self.squeeze=nn.Linear(sum(s.channels for s in specs),hidden)
        self.excite=nn.ModuleList([nn.Linear(hidden,s.channels) for s in specs])
        for layer in self.excite:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.metadata.update(reduction_ratio=reduction_ratio,hidden_dimension=hidden,
            adaptation='single stage; paper 2*sigmoid; zero excitation weight and bias for identity initialization')
        self.finish_metadata()

    def forward(self,*features):
        self.check(features)
        shared=torch.relu(self.squeeze(torch.cat([x.flatten(2).mean(-1) for x in features],1)))
        deltas=[x*(2*torch.sigmoid(layer(shared))-1).reshape(x.shape[0],x.shape[1],*([1]*(x.ndim-2))) for x,layer in zip(features,self.excite)]
        return self.packet(features,deltas)


class AttentionDirection(nn.Module):
    def __init__(self,source_channels,destination_channels,dimension,heads):
        super().__init__();self.heads=heads;self.dimension=dimension
        self.query=nn.Linear(destination_channels,dimension)
        self.key=nn.Linear(source_channels,dimension)
        self.value=nn.Linear(source_channels,dimension)
        self.output=nn.Linear(dimension,destination_channels)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,destination,source):
        n=destination.shape[0];h=self.heads;d=self.dimension//h
        q=self.query(destination).reshape(n,-1,h,d).transpose(1,2)
        k=self.key(source).reshape(n,-1,h,d).transpose(1,2)
        v=self.value(source).reshape(n,-1,h,d).transpose(1,2)
        # Explicit definition avoids backend-dependent dropout or fused kernels.
        z=(torch.softmax(q@k.transpose(-1,-2)/math.sqrt(d),dim=-1)@v).transpose(1,2).reshape(n,-1,self.dimension)
        return self.output(z)


class AttentionExchange(NativeExchange):
    def __init__(self,specs,attention_dimension,heads):
        super().__init__(specs,'cross_attention')
        positive_integer(attention_dimension,'attention_dimension',1);positive_integer(heads,'heads',1)
        if attention_dimension%heads:raise ValueError('Attention dimension must divide into heads')
        self.norms=nn.ModuleList([nn.LayerNorm(s.channels) for s in specs])
        self.directions=nn.ModuleList([AttentionDirection(specs[1-i].channels,specs[i].channels,attention_dimension,heads) for i in range(2)])
        self.metadata.update(attention_dimension=attention_dimension,heads=heads,pre_layernorm=True,
            direction_order=[[self.keys[1-i],self.keys[i]] for i in range(2)],
            projection_bias=True,dropout=0.,feed_forward=False,explicit_position_encoding=False,
            adaptation='full spatial tokens; simultaneous bidirectional attention; zero output projections')
        self.finish_metadata()

    def forward(self,*features):
        self.check(features)
        tokens=[norm(x.flatten(2).transpose(1,2)) for norm,x in zip(self.norms,features)]
        deltas=[layer(tokens[i],tokens[1-i]).transpose(1,2).reshape_as(features[i]) for i,layer in enumerate(self.directions)]
        return self.packet(features,deltas)
