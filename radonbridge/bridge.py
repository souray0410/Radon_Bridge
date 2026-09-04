"""Ratio-configured, native-node-preserving many-participant Rhythm Bridge."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .projector import Projector, ScrambledProjector, GaussianProjector, positive_integer

@dataclass(frozen=True)
class FeatureSpec:
    key: str
    channels: int
    shape: tuple

class LinearMixer(nn.Module):
    def __init__(self,widths,kernel_size=3,self_only=False):
        super().__init__();self.widths=tuple(widths)
        if kernel_size<1 or kernel_size%2!=1:raise ValueError("Use a positive odd kernel")
        n=sum(widths);self.conv=nn.Conv1d(n,n,kernel_size,padding=kernel_size//2,bias=False)
        nn.init.zeros_(self.conv.weight)
        mask=torch.ones_like(self.conv.weight)
        if self_only:
            mask.zero_();start=0
            for width in widths:mask[start:start+width,start:start+width]=1;start+=width
        self.register_buffer("mask",mask)
    def forward(self,*features):
        z=torch.cat(features,dim=1)
        y=nn.functional.conv1d(z,self.conv.weight*self.mask,padding=self.conv.kernel_size[0]//2)
        return torch.split(y,self.widths,dim=1)

class BridgeExchange(nn.Module):
    def __init__(self, specs, *, M, S, rho, mode='radon'):
        super().__init__()
        for name, value, minimum in [('M', M, 1), ('S', S, 2)]:
            positive_integer(value, name, minimum)
        if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError('rho must be a finite compression ratio in (0,1]')
        if mode not in ('radon', 'self', 'pooled', 'random', 'scrambled'):
            raise ValueError(mode)
        self.keys = [s.key for s in specs]
        self.shapes = [(s.channels, *s.shape) for s in specs]
        self.lengths = [math.prod(s) for s in self.shapes]
        self.mode, self.M, self.S, self.rho = mode, M, S, rho
        kind={'random':GaussianProjector,'scrambled':ScrambledProjector}.get(mode,Projector)
        self.projectors = nn.ModuleList([] if mode == 'pooled' else [
            kind(s.shape,M,S,seed=int(torch.initial_seed())+1009*i+len(s.shape)) if mode in ('random','scrambled') else kind(s.shape,M,S)
            for i,s in enumerate(specs)])
        widths = [s.channels if mode == 'pooled' else s.channels*M for s in specs]
        retained = [max(1, math.floor(rho*w)) for w in widths]
        self.compress = nn.ModuleList([nn.Conv1d(w, h, 1, bias=False) for w,h in zip(widths,retained)])
        self.expand = nn.ModuleList([nn.Conv1d(h, w, 1, bias=False) for w,h in zip(widths,retained)])
        self.mixer = LinearMixer(retained, 1 if mode == 'pooled' else 3, mode == 'self')
        self.metadata = {'mode': mode, 'M': M, 'S': S, 'rho': rho, 'participants': [
            {'key': s.key, 'channels': s.channels, 'shape': list(s.shape), 'projected_channels': w,
             'retained_channels': retained[i], 'achieved_width_ratio': retained[i]/w,
             'geometry': self.projectors[i].metadata if mode != 'pooled' else None}
            for i, (s, w) in enumerate(zip(specs, widths))]}
        self.latest_inputs = self.latest_deltas = None

    def forward(self, *features):
        if len(features) != len(self.shapes) or any(tuple(x.shape[1:]) != shape for x, shape in zip(features, self.shapes)):
            raise ValueError('Participant shapes changed')
        if len({x.shape[0] for x in features}) != 1:
            raise ValueError('Participants must share batch alignment')
        projected = [x.flatten(2).mean(-1, keepdim=True) if self.mode == 'pooled' else self.projectors[i](x)
                     for i, x in enumerate(features)]
        encoded = [layer(p) for layer, p in zip(self.compress, projected)]
        mixed = self.mixer(*encoded)
        expanded = [layer(p) for layer, p in zip(self.expand, mixed)]
        deltas = [p.reshape(p.shape[0], shape[0], *([1]*(len(shape)-1))).expand_as(x)
                  if self.mode == 'pooled' else self.projectors[i].backproject(p)
                  for i, (p, x, shape) in enumerate(zip(expanded, features, self.shapes))]
        # Diagnostics retain detached views, not the training autograd graph.
        self.latest_inputs = tuple(x.detach() for x in features)
        self.latest_deltas = tuple(x.detach() for x in deltas)
        return torch.cat([(x+delta).flatten(1) for x, delta in zip(features, deltas)], dim=1)

class ReturnParticipant(nn.Module):
    """Parameter-free routing back to an existing native Node."""
    def __init__(self, start, length, shape):
        super().__init__(); self.start=start; self.length=length; self.shape=tuple(shape)
    def forward(self, packet):
        return packet[:, self.start:self.start+self.length].reshape(packet.shape[0], *self.shape)


def attach_group(node, edge, specs, inputs, prefix, *, M, S, rho, mode='radon'):
    if not specs or len({s.key for s in specs}) != len(specs) or set(inputs) != {s.key for s in specs}:
        raise ValueError('Participant identity mismatch')
    exchange = BridgeExchange(specs, M=M, S=S, rho=rho, mode=mode)
    packet = node(prefix+'communication')
    edge(prefix+'exchange', exchange, [inputs[s.key] for s in specs], [packet])
    start = 0
    for spec, shape, length in zip(specs, exchange.shapes, exchange.lengths):
        edge(prefix+spec.key+'_return', ReturnParticipant(start, length, shape), [packet], [inputs[spec.key]],
             level_group=prefix+'return_level')
        start += length
    meta = dict(exchange.metadata, topology='level_inplace', communication_node=packet,
                native_node_ids=dict(inputs), exchange_edge_name=prefix+'exchange')
    return dict(inputs), meta


def attach_to_nodes(builder, node_names, *, prefix, M, S, rho, samples=None, mode='radon'):
    if len(set(node_names)) != len(node_names) or not node_names:
        raise ValueError('Select distinct existing nodes')
    specs, inputs = [], {}
    for name in node_names:
        native = builder.by_name[name]
        feature = samples[name] if samples is not None else native.feature_message.current_state
        if not isinstance(feature, torch.Tensor) or feature.ndim < 3:
            raise ValueError('Expected [batch,channels,*spatial]')
        specs.append(FeatureSpec(name, feature.shape[1], tuple(feature.shape[2:])))
        inputs[name] = native.id
    old=list(builder.steps)
    last_produced={nid:i for i,(_,_,tails) in enumerate(old) for nid in tails}
    affected={nid:last_produced.get(nid,-1) for nid in inputs.values()}
    delayed=set()
    for i,(_,heads,tails) in enumerate(old):
        if any(n in affected and i>affected[n] for n in heads):
            delayed.add(i)
            for n in tails: affected[n]=min(affected.get(n,i),i)
    if any(last_produced.get(n,-1) in delayed for n in inputs.values()):
        raise ValueError('Selected Nodes are causally nested; choose one frontier per network or specify a versioned iterative schedule')
    result,meta=attach_group(builder.node,builder.edge,specs,inputs,prefix,M=M,S=S,rho=rho,mode=mode)
    inserted=builder.steps[len(old):]
    builder.steps[:]=[row for i,row in enumerate(old) if i not in delayed]+inserted+[row for i,row in enumerate(old) if i in delayed]
    meta.update(shape_inference='representative_features',native_edges_rewired=False,
                deferred_native_edges=[old[i][0] for i in sorted(delayed)])
    return result,meta
