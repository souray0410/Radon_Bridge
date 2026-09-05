"""Ratio-configured, native-node-preserving many-participant Radon Bridge."""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import torch
from torch import nn
from .learned_channel import LearnedChannelCodec
from .svd_basis import FixedChannelBasis, BASIS_VERSION, QR_VERSION, CENTERED_VERSION
from .projector import Projector, ScrambledProjector, GaussianProjector, LinearResampleProjector, positive_integer

@dataclass(frozen=True)
class FeatureSpec:
    key: str
    channels: int
    shape: tuple


def participant_values(value, keys, name):
    """A scalar broadcasts; a mapping must identify every participant exactly."""
    if isinstance(value, Mapping):
        if set(value) != set(keys):
            raise ValueError(f'{name} keys must match participant keys exactly')
        return [value[key] for key in keys]
    return [value for _ in keys]

class LinearMixer(nn.Module):
    def __init__(self,widths,kernel_size=3,self_only=False,keys=None,cross_edges=None):
        super().__init__();self.widths=tuple(widths)
        if kernel_size<1 or kernel_size%2!=1:raise ValueError("Use a positive odd kernel")
        n=sum(widths);self.conv=nn.Conv1d(n,n,kernel_size,padding=kernel_size//2,bias=False)
        nn.init.zeros_(self.conv.weight)
        mask=torch.ones_like(self.conv.weight)
        if self_only:
            mask.zero_();start=0
            for width in widths:mask[start:start+width,start:start+width]=1;start+=width
        if cross_edges is not None:
            if self_only:raise ValueError('Do not combine self mode and cross_edges')
            if keys is None or len(keys)!=len(widths):raise ValueError('Named sources required')
            edges=[]
            for pair in cross_edges:
                if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError('cross_edges entries are [source,destination]')
                source,destination=pair
                if source not in keys or destination not in keys or source==destination:raise ValueError('Invalid cross edge')
                edges.append((source,destination))
            if len(set(edges))!=len(edges):raise ValueError('Duplicate cross edge')
            offsets=[0]
            for w in widths:offsets.append(offsets[-1]+w)
            for dst in range(len(widths)):
                for src in range(len(widths)):
                    if dst!=src and (keys[src],keys[dst]) not in edges:
                        mask[offsets[dst]:offsets[dst+1],offsets[src]:offsets[src+1]]=0
        self.register_buffer("mask",mask)
    def forward(self,*features):
        z=torch.cat(features,dim=1)
        y=nn.functional.conv1d(z,self.conv.weight*self.mask,padding=self.conv.kernel_size[0]//2)
        return torch.split(y,self.widths,dim=1)

class BridgeExchange(nn.Module):
    def __init__(self, specs, *, M, S, rho, mode='radon', compression='learned_projected', basis_files=None,cross_edges=None):
        super().__init__()
        self.keys = [s.key for s in specs]
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError('Distinct nonempty participant keys required')
        directions = participant_values(M, self.keys, 'M')
        ratios = participant_values(rho, self.keys, 'rho')
        positive_integer(S, 'S', 2)
        for value in directions:
            positive_integer(value, 'M', 1)
        for value in ratios:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('rho must be a finite compression ratio in (0,1]')
        if mode not in ('radon', 'self', 'pooled', 'random', 'scrambled', 'linear_resample'):
            raise ValueError(mode)
        if compression not in ('learned_projected','fixed_svd_channel','fixed_random_orthogonal_channel','fixed_centered_svd_channel','learned_channel'):
            raise ValueError('Unknown compression method')
        if compression in ('fixed_svd_channel','fixed_centered_svd_channel') and mode not in ('radon','self','scrambled','linear_resample'):
            raise ValueError('Unsupported SVD mechanism')
        if compression=='fixed_random_orthogonal_channel' and mode not in ('radon','linear_resample'):
            raise ValueError('Random orthogonal channel supports Radon and linear resampling')
        if compression=='learned_projected' and mode=='linear_resample':
            raise ValueError('Linear resampling is a fixed SVD control')
        if compression in ('learned_projected','learned_channel') and basis_files is not None:
            raise ValueError('basis_files is only valid for fixed channel compression')
        self.compression=compression
        self.shapes = [(s.channels, *s.shape) for s in specs]
        self.lengths = [math.prod(s) for s in self.shapes]
        self.mode, self.M, self.S, self.rho = mode, M, S, rho
        kind={'random':GaussianProjector,'scrambled':ScrambledProjector,'linear_resample':LinearResampleProjector}.get(mode,Projector)
        self.projectors = nn.ModuleList([] if mode == 'pooled' else [
            kind(s.shape,directions[i],S,seed=int(torch.initial_seed())+1009*i+len(s.shape)) if mode in ('random','scrambled') else kind(s.shape,directions[i],S)
            for i,s in enumerate(specs)])
        widths = [s.channels if mode == 'pooled' else s.channels*directions[i] for i,s in enumerate(specs)]
        retained = [max(1, math.floor(ratio*w)) for ratio,w in zip(ratios,widths)]
        if compression=='learned_projected':
            self.compress = nn.ModuleList([nn.Conv1d(w, h, 1, bias=False) for w,h in zip(widths,retained)])
            self.expand = nn.ModuleList([nn.Conv1d(h, w, 1, bias=False) for w,h in zip(widths,retained)])
        elif compression=='learned_channel':
            if mode not in ('radon','self','scrambled','linear_resample'):raise ValueError('Unsupported learned channel mechanism')
            ranks=[max(1,math.floor(ratio*s.channels)) for ratio,s in zip(ratios,specs)]
            self.channel_codecs=nn.ModuleList([LearnedChannelCodec(s.channels,rank,s.key,int(torch.initial_seed())) for s,rank in zip(specs,ranks)])
            retained=[rank*m for rank,m in zip(ranks,directions)]
        else:
            if not isinstance(basis_files,dict) or set(basis_files)!=set(self.keys):
                raise ValueError('basis_files must match participant keys exactly')
            ranks=[max(1,math.floor(ratio*s.channels)) for ratio,s in zip(ratios,specs)]
            self.channel_bases=nn.ModuleList([FixedChannelBasis(s.channels,rank,s.key,basis_files[s.key],version=QR_VERSION if compression=='fixed_random_orthogonal_channel' else CENTERED_VERSION if compression=='fixed_centered_svd_channel' else BASIS_VERSION) for s,rank in zip(specs,ranks)])
            retained=[rank*m for rank,m in zip(ranks,directions)]
        self.mixer = LinearMixer(retained, 1 if mode == 'pooled' else 3, mode == 'self',self.keys,cross_edges)
        self.metadata = {'mode': mode, 'M': M, 'S': S, 'rho': rho, 'participants': [
            {'key': s.key, 'channels': s.channels, 'shape': list(s.shape), 'M': directions[i], 'rho': ratios[i], 'projected_channels': w,
             'retained_channels': retained[i], 'achieved_width_ratio': retained[i]/w,
             'geometry': self.projectors[i].metadata if mode != 'pooled' else None}
            for i, (s, w) in enumerate(zip(specs, widths))]}
        self.metadata['compression']=compression
        if cross_edges is not None:self.metadata.update(cross_edges=cross_edges,cross_edges_convention='[source,destination]; weight rows=destination, columns=source')
        self.metadata['stored_bridge_parameters']=sum(p.numel() for p in self.parameters())
        self.metadata['effective_bridge_parameters']=self.metadata['stored_bridge_parameters']-self.mixer.conv.weight.numel()+int(self.mixer.mask.sum())
        if compression=='learned_channel':
            for participant,codec in zip(self.metadata['participants'],self.channel_codecs):
                participant.update(channel_rank=codec.metadata['channel_rank'],effective_channel_ratio=codec.metadata['channel_rank']/codec.metadata['channels'],channel_codec=codec.metadata)
        elif compression!='learned_projected':
            for participant,basis in zip(self.metadata['participants'],self.channel_bases):
                participant['channel_rank']=basis.q.shape[1]
                participant['effective_channel_ratio']=basis.q.shape[1]/basis.q.shape[0]
                participant['basis']=dict(basis.metadata)
        self.latest_inputs = self.latest_deltas = None

    def export_fixed_bases(self, directory):
        if self.compression in ('learned_projected','learned_channel'):return []
        artifacts=[basis.export(directory) for basis in self.channel_bases]
        for participant,artifact in zip(self.metadata['participants'],artifacts):participant['basis']=artifact
        return artifacts

    def forward(self, *features):
        if len(features) != len(self.shapes) or any(tuple(x.shape[1:]) != shape for x, shape in zip(features, self.shapes)):
            raise ValueError('Participant shapes changed')
        if len({x.shape[0] for x in features}) != 1:
            raise ValueError('Participants must share batch alignment')
        if self.compression=='learned_channel':
            encoded=[projector(codec.encode(x)) for codec,projector,x in zip(self.channel_codecs,self.projectors,features)]
            mixed=self.mixer(*encoded)
            deltas=[codec.decode(projector.backproject(p)) for codec,projector,p in zip(self.channel_codecs,self.projectors,mixed)]
        elif self.compression!='learned_projected':
            encoded=[projector(basis.encode(x)) for basis,projector,x in zip(self.channel_bases,self.projectors,features)]
            mixed=self.mixer(*encoded)
            deltas=[basis.decode(projector.backproject(p)) for basis,projector,p in zip(self.channel_bases,self.projectors,mixed)]
        else:
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


def attach_group(node, edge, specs, inputs, prefix, *, M=None, S=None, rho=None, mode='radon', compression='learned_projected', basis_files=None,cross_edges=None,family='radon',reduction_ratio=None,attention_dimension=None,heads=None):
    if not specs or len({s.key for s in specs}) != len(specs) or set(inputs) != {s.key for s in specs}:
        raise ValueError('Participant identity mismatch')
    if family=='radon':
        if any(v is not None for v in (reduction_ratio,attention_dimension,heads)):raise ValueError('Baseline-only fields supplied to Radon')
        exchange = BridgeExchange(specs, M=M, S=S, rho=rho, mode=mode, compression=compression, basis_files=basis_files,cross_edges=cross_edges)
    else:
        if any(v is not None for v in (M,S,rho,basis_files,cross_edges)) or mode!='radon' or compression!='learned_projected':raise ValueError('Radon-only fields supplied to baseline')
        from .baselines import MMTMExchange,AttentionExchange
        if family=='mmtm' and attention_dimension is None and heads is None:exchange=MMTMExchange(specs,reduction_ratio)
        elif family=='cross_attention' and reduction_ratio is None:exchange=AttentionExchange(specs,attention_dimension,heads)
        else:raise ValueError('Unknown family or incompatible configuration')
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


def attach_to_nodes(builder, node_names, *, prefix, M=None, S=None, rho=None, samples=None, mode='radon', compression='learned_projected', basis_files=None,cross_edges=None,family='radon',reduction_ratio=None,attention_dimension=None,heads=None):
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
    for i,(_,read_heads,tails) in enumerate(old):
        if any(n in affected and i>affected[n] for n in read_heads):
            delayed.add(i)
            for n in tails: affected[n]=min(affected.get(n,i),i)
    if any(last_produced.get(n,-1) in delayed for n in inputs.values()):
        raise ValueError('Selected Nodes are causally nested; choose one frontier per network or specify a versioned iterative schedule')
    result,meta=attach_group(builder.node,builder.edge,specs,inputs,prefix,M=M,S=S,rho=rho,mode=mode,compression=compression,basis_files=basis_files,cross_edges=cross_edges,family=family,reduction_ratio=reduction_ratio,attention_dimension=attention_dimension,heads=heads)
    inserted=builder.steps[len(old):]
    builder.steps[:]=[row for i,row in enumerate(old) if i not in delayed]+inserted+[row for i,row in enumerate(old) if i in delayed]
    meta.update(shape_inference='representative_features',native_edges_rewired=False,
                deferred_native_edges=[old[i][0] for i in sorted(delayed)])
    return result,meta
