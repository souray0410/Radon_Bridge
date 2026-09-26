"""Ratio-configured, native-node-preserving many-participant Radon Bridge."""
from dataclasses import dataclass
from collections.abc import Mapping
import math
import torch
from torch import nn
from radon_bridge.methods.channel import LearnedChannelCodec
from radon_bridge.methods.factorized import FactorizedMixer
from radon_bridge.methods.basis import FixedChannelBasis, BASIS_VERSION, QR_VERSION, CENTERED_VERSION
from radon_bridge.methods.projection import Projector, ScrambledProjector, GaussianProjector, LinearResampleProjector, positive_integer

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
    def __init__(self,widths,kernel_size=3,self_only=False,keys=None,cross_edges=None,s_axis_permutation=None,group_count=1,source_ranks=None,directions=None):
        super().__init__();self.widths=tuple(widths)
        if kernel_size<1 or kernel_size%2!=1:raise ValueError("Use a positive odd kernel")
        positive_integer(group_count,'group_count')
        if group_count>1:
            if self_only or cross_edges is not None or s_axis_permutation is not None:
                raise ValueError('Grouped mixing cannot combine with self, cross-edge or S-axis controls')
            if source_ranks is None or directions is None or len(source_ranks)!=len(widths) or len(directions)!=len(widths):
                raise ValueError('Grouped mixing requires one rank and direction count per source')
            if any(rank%group_count for rank in source_ranks):
                raise ValueError('Each source channel rank must be divisible by group_count')
            if any(width!=rank*m for width,rank,m in zip(widths,source_ranks,directions)):
                raise ValueError('Grouped source width must equal channel rank times directions')
        n=sum(widths);self.conv=nn.Conv1d(n,n,kernel_size,padding=kernel_size//2,bias=False,groups=group_count)
        nn.init.zeros_(self.conv.weight)
        mask=torch.ones_like(self.conv.weight)
        if group_count>1:
            offsets=[0]
            for width in widths:offsets.append(offsets[-1]+width)
            order=[]
            for group in range(group_count):
                for offset,rank,m in zip(offsets[:-1],source_ranks,directions):
                    block=rank//group_count
                    for channel in range(group*block,(group+1)*block):
                        order.extend(offset+channel*m+direction for direction in range(m))
            permutation=torch.tensor(order,dtype=torch.long)
            if len(order)!=n or len(set(order))!=n:raise RuntimeError('Invalid grouped channel permutation')
            self.register_buffer('group_permutation',permutation)
            self.register_buffer('group_inverse',torch.argsort(permutation))
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
        if s_axis_permutation is not None:
            from radon_bridge.methods.sampling import validate_permutation
            values=validate_permutation(s_axis_permutation)
            self.register_buffer('s_axis_permutation',torch.tensor(values,dtype=torch.long))
            self.register_buffer('s_axis_inverse',torch.argsort(self.s_axis_permutation))
    def _load_from_state_dict(self,state_dict,prefix,*args,**kwargs):
        # A checkpoint must not silently replace a prespecified structural control.
        for name in ('s_axis_permutation','s_axis_inverse','group_permutation','group_inverse'):
            if hasattr(self,name) and prefix+name in state_dict:
                if not torch.equal(state_dict[prefix+name].cpu(),getattr(self,name).cpu()):
                    raise RuntimeError('Checkpoint structural permutation does not match configuration')
        return super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)
    def forward(self,*features):
        z=torch.cat(features,dim=1)
        if hasattr(self,'group_permutation'):z=z.index_select(1,self.group_permutation)
        if hasattr(self,'s_axis_permutation'):
            if z.shape[-1]!=len(self.s_axis_permutation):raise ValueError('S-axis length mismatch')
            z=z.index_select(-1,self.s_axis_permutation)
        y=nn.functional.conv1d(z,self.conv.weight*self.mask,padding=self.conv.kernel_size[0]//2,groups=self.conv.groups)
        if hasattr(self,'s_axis_inverse'):y=y.index_select(-1,self.s_axis_inverse)
        if hasattr(self,'group_inverse'):y=y.index_select(1,self.group_inverse)
        return torch.split(y,self.widths,dim=1)

class BridgeExchange(nn.Module):
    def __init__(self, specs, *, M, S, rho, mode='radon', compression='learned_projected', basis_files=None,cross_edges=None,nested_rhos=None,s_axis_permutation=None,kernel_size=3,r=None,h=None,group_count=1,bottleneck_rank=None):
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
        if compression not in ('factorized_projected','learned_projected','fixed_svd_channel','fixed_random_orthogonal_channel','fixed_centered_svd_channel','learned_channel'):
            raise ValueError('Unknown compression method')
        if compression == 'factorized_projected':
            if (mode not in ('radon','linear_resample') or rho != 1 or group_count != 1
                    or any(v is not None for v in (basis_files,cross_edges,nested_rhos,s_axis_permutation,r,h))):
                raise ValueError('Global factorization requires full projected input and ungrouped Radon/resampling')
            positive_integer(bottleneck_rank,'bottleneck_rank')
        elif bottleneck_rank is not None:
            raise ValueError('bottleneck_rank belongs only to global factorization')
        if compression in ('fixed_svd_channel','fixed_centered_svd_channel') and mode not in ('radon','self','scrambled','linear_resample'):
            raise ValueError('Unsupported SVD mechanism')
        if compression=='fixed_random_orthogonal_channel' and mode not in ('radon','self','scrambled','linear_resample'):
            raise ValueError('Unsupported random orthogonal channel mechanism')
        if compression=='learned_projected' and mode=='linear_resample':
            raise ValueError('Linear resampling is a fixed SVD control')
        if compression in ('learned_projected','learned_channel','factorized_projected') and basis_files is not None:
            raise ValueError('basis_files is only valid for fixed channel compression')
        if s_axis_permutation is not None:
            from radon_bridge.methods.sampling import validate_permutation
            validate_permutation(s_axis_permutation,S)
            if mode!='radon' or compression not in ('fixed_svd_channel','fixed_random_orthogonal_channel') or nested_rhos is not None or cross_edges is not None:
                raise ValueError('S-axis control requires independent-width bidirectional fixed SVD/QR Radon')
        positive_integer(kernel_size, 'kernel_size', 1)
        if kernel_size % 2 != 1: raise ValueError('kernel_size must be odd')
        if mode == 'pooled' and kernel_size != 3: raise ValueError('Legacy pooled kernel is fixed at one')
        if nested_rhos is not None and kernel_size != 3: raise ValueError('Joint-width legacy protocol uses kernel three')
        positive_integer(group_count,'group_count')
        if group_count>1 and (compression!='fixed_svd_channel' or mode not in ('radon','linear_resample') or cross_edges is not None or nested_rhos is not None or s_axis_permutation is not None):
            raise ValueError('Grouped protocol requires fixed SVD Radon/linear-resample without other structural controls')
        if r is not None or h is not None:
            if compression not in ('fixed_svd_channel', 'fixed_random_orthogonal_channel') or r is None or h is None:
                raise ValueError('Explicit r/h requires fixed SVD/QR and both fields')
            ranks_requested = participant_values(r, self.keys, 'r')
            widths_requested = participant_values(h, self.keys, 'h')
            for spec, m, ratio, rank, width in zip(specs, directions, ratios, ranks_requested, widths_requested):
                positive_integer(rank, 'r', 1); positive_integer(width, 'h', 1)
                if rank > spec.channels or width != rank*m or ratio != rank/spec.channels:
                    raise ValueError('Inconsistent r, h, M, rho, C')
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
        if compression=='factorized_projected':
            self.compress=nn.ModuleList([nn.Identity() for _ in widths])
            self.expand=nn.ModuleList([nn.Identity() for _ in widths])
        elif compression=='learned_projected':
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
        self.mixer = (FactorizedMixer(widths,bottleneck_rank,kernel_size) if compression=='factorized_projected'
                      else LinearMixer(retained, 1 if mode == 'pooled' else kernel_size, mode == 'self',self.keys,cross_edges,s_axis_permutation,
                                       group_count=group_count,source_ranks=ranks if group_count>1 else None,directions=directions if group_count>1 else None))
        self.metadata = {'mode': mode, 'M': M, 'S': S, 'rho': rho, 'participants': [
            {'key': s.key, 'channels': s.channels, 'shape': list(s.shape), 'M': directions[i], 'rho': ratios[i], 'projected_channels': w,
             'retained_channels': retained[i], 'achieved_width_ratio': retained[i]/w,
             'geometry': self.projectors[i].metadata if mode != 'pooled' else None}
            for i, (s, w) in enumerate(zip(specs, widths))]}
        self.metadata['compression']=compression
        if compression=='factorized_projected':
            self.metadata.update(bottleneck_rank=bottleneck_rank,factorization='global_B_K_A',
                                 fixed_channel_compression=False,global_projected_width=sum(widths),
                                 effective_kernel_rank_bound=bottleneck_rank)
        self.metadata['kernel_size']=self.mixer.conv.kernel_size[0]
        if group_count>1:
            self.metadata['group_count']=group_count
            self.metadata['grouping_order']='group,source,channel,direction; inverse restored before source split'
            dense_mixer_parameters=sum(retained)**2*self.mixer.conv.kernel_size[0]
            self.metadata['dense_equivalent_mixer_parameters']=dense_mixer_parameters
            self.metadata['grouped_mixer_parameters']=self.mixer.conv.weight.numel()
            self.metadata['grouped_connection_fraction']=self.mixer.conv.weight.numel()/dense_mixer_parameters
        if s_axis_permutation is not None:
            from radon_bridge.methods.sampling import permutation_metadata
            self.metadata['s_axis_control']=permutation_metadata(s_axis_permutation)
        if cross_edges is not None:self.metadata.update(cross_edges=cross_edges,cross_edges_convention='[source,destination]; weight rows=destination, columns=source')
        self.metadata['stored_bridge_parameters']=sum(p.numel() for p in self.parameters())
        self.metadata['effective_bridge_parameters']=self.metadata['stored_bridge_parameters']-self.mixer.conv.weight.numel()+int(self.mixer.mask.sum())
        if compression=='learned_channel':
            for participant,codec in zip(self.metadata['participants'],self.channel_codecs):
                participant.update(channel_rank=codec.metadata['channel_rank'],effective_channel_ratio=codec.metadata['channel_rank']/codec.metadata['channels'],channel_codec=codec.metadata)
        elif compression not in ('learned_projected','factorized_projected'):
            for participant,basis in zip(self.metadata['participants'],self.channel_bases):
                participant['channel_rank']=basis.q.shape[1]
                participant['effective_channel_ratio']=basis.q.shape[1]/basis.q.shape[0]
                participant['basis']=dict(basis.metadata)
        self.nested_rhos = None
        self.active_rho = rho
        if nested_rhos is not None:
            if (mode!='radon' or cross_edges is not None or
                compression not in ('learned_channel','fixed_svd_channel','fixed_random_orthogonal_channel')):
                raise ValueError('Nested widths require standard channel Radon')
            if not isinstance(nested_rhos,(list,tuple)) or not nested_rhos or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=1 for v in nested_rhos):
                raise ValueError('Invalid nested ratios')
            if list(nested_rhos)!=sorted(set(nested_rhos)) or not isinstance(rho,(int,float)) or rho!=max(nested_rhos):
                raise ValueError('Allocate maximum rho and supply increasing unique widths')
            self.nested_rhos=tuple(nested_rhos)
            self.metadata['nested_rhos']=list(nested_rhos)
            self.metadata['nested_indexing']='source-major, channel-prefix, all M directions; shared maximum mixer submatrix'
        self.latest_inputs = self.latest_deltas = None

    def export_fixed_bases(self, directory):
        if self.compression in ('learned_projected','learned_channel','factorized_projected'):return []
        artifacts=[basis.export(directory) for basis in self.channel_bases]
        for participant,artifact in zip(self.metadata['participants'],artifacts):participant['basis']=artifact
        return artifacts

    def set_rho(self,rho):
        if self.nested_rhos is None or rho not in self.nested_rhos:
            raise ValueError('Width is not in the declared nested set')
        self.active_rho=rho

    def nested_delta(self,features):
        ranks=[max(1,math.floor(self.active_rho*shape[0])) for shape in self.shapes]
        encoded=[]
        for i,(x,r,projector) in enumerate(zip(features,ranks,self.projectors)):
            if self.compression=='learned_channel':
                w=self.channel_codecs[i].encoder.weight[:r]
                z=nn.functional.conv1d(x.flatten(2),w,bias=None).reshape(x.shape[0],r,*x.shape[2:])
            else:
                q=self.channel_bases[i].q[:,:r].contiguous()
                z=torch.einsum('cr,bc...->br...',q,x)
            encoded.append(projector(z))
        widths=[z.shape[1] for z in encoded]
        if tuple(widths)==self.mixer.widths:
            mixed=self.mixer(*encoded)
        else:
            offsets=[0]
            for w in self.mixer.widths:offsets.append(offsets[-1]+w)
            ix=torch.cat([torch.arange(offset,offset+w,device=features[0].device) for offset,w in zip(offsets,widths)])
            w=self.mixer.conv.weight.index_select(0,ix).index_select(1,ix)
            mask=self.mixer.mask.index_select(0,ix).index_select(1,ix)
            y=nn.functional.conv1d(torch.cat(encoded,1),w*mask,padding=1)
            mixed=y.split(widths,dim=1)
        deltas=[]
        for i,(x,r,projector,z) in enumerate(zip(features,ranks,self.projectors,mixed)):
            returned=projector.backproject(z)
            if self.compression=='learned_channel':
                delta=nn.functional.conv1d(returned.flatten(2),self.channel_codecs[i].decoder.weight[:,:r],bias=None).reshape_as(x)
            else:
                delta=torch.einsum('cr,br...->bc...',self.channel_bases[i].q[:,:r].contiguous(),returned)
            deltas.append(delta)
        return deltas

    def forward(self, *features):
        if len(features) != len(self.shapes) or any(tuple(x.shape[1:]) != shape for x, shape in zip(features, self.shapes)):
            raise ValueError('Participant shapes changed')
        if len({x.shape[0] for x in features}) != 1:
            raise ValueError('Participants must share batch alignment')
        if self.nested_rhos is not None:
            deltas=self.nested_delta(features)
        elif self.compression=='learned_channel':
            encoded=[projector(codec.encode(x)) for codec,projector,x in zip(self.channel_codecs,self.projectors,features)]
            mixed=self.mixer(*encoded)
            deltas=[codec.decode(projector.backproject(p)) for codec,projector,p in zip(self.channel_codecs,self.projectors,mixed)]
        elif self.compression not in ('learned_projected','factorized_projected'):
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
        if getattr(self, 'delta_only', False):
            return torch.cat([delta.flatten(1) for delta in deltas], dim=1)
        return torch.cat([(x+delta).flatten(1) for x, delta in zip(features, deltas)], dim=1)

class ReturnParticipant(nn.Module):
    """Parameter-free routing back to an existing native Node."""
    def __init__(self, start, length, shape):
        super().__init__(); self.start=start; self.length=length; self.shape=tuple(shape)
    def forward(self, packet):
        return packet[:, self.start:self.start+self.length].reshape(packet.shape[0], *self.shape)


def attach_group(node, edge, specs, inputs, prefix, *, M=None, S=None, rho=None, mode='radon', compression='learned_projected', basis_files=None,cross_edges=None,family='radon',reduction_ratio=None,attention_dimension=None,heads=None,alignment_tokens=None,nested_rhos=None,s_axis_permutation=None,kernel_size=3,r=None,h=None,group_count=1,bottleneck_rank=None):
    if not specs or len({s.key for s in specs}) != len(specs) or set(inputs) != {s.key for s in specs}:
        raise ValueError('Participant identity mismatch')
    if family=='radon':
        if any(v is not None for v in (reduction_ratio,attention_dimension,heads,alignment_tokens)):raise ValueError('Baseline-only fields supplied to Radon')
        exchange = BridgeExchange(specs, M=M, S=S, rho=rho, mode=mode, compression=compression, basis_files=basis_files,cross_edges=cross_edges,nested_rhos=nested_rhos,s_axis_permutation=s_axis_permutation,kernel_size=kernel_size,r=r,h=h,group_count=group_count,bottleneck_rank=bottleneck_rank)
    else:
        if any(v is not None for v in (M,S,rho,basis_files,cross_edges,nested_rhos,s_axis_permutation,r,h,bottleneck_rank)) or group_count!=1 or mode!='radon' or compression!='learned_projected':raise ValueError('Radon-only fields supplied to baseline')
        if kernel_size != 3: raise ValueError('Radon kernel field is not applicable to a nonlinear baseline')
        from radon_bridge.methods.baselines import MMTMExchange, AuthorMMTMExchange, AttentionExchange, CMXRectifyExchange
        if family=='mmtm' and attention_dimension is None and heads is None and alignment_tokens is None:exchange=MMTMExchange(specs,reduction_ratio)
        elif family=='mmtm_author' and attention_dimension is None and heads is None and alignment_tokens is None:exchange=AuthorMMTMExchange(specs,reduction_ratio)
        elif family=='cross_attention' and reduction_ratio is None and alignment_tokens is None:exchange=AttentionExchange(specs,attention_dimension,heads)
        elif family=='cmx_frm' and reduction_ratio is None and attention_dimension is None and heads is None:exchange=CMXRectifyExchange(specs,alignment_tokens)
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


def attach_to_nodes(builder, node_names, *, prefix, M=None, S=None, rho=None, samples=None, mode='radon', compression='learned_projected', basis_files=None,cross_edges=None,family='radon',reduction_ratio=None,attention_dimension=None,heads=None,alignment_tokens=None,nested_rhos=None,s_axis_permutation=None,kernel_size=3,r=None,h=None,group_count=1,bottleneck_rank=None):
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
    result,meta=attach_group(builder.node,builder.edge,specs,inputs,prefix,M=M,S=S,rho=rho,mode=mode,compression=compression,basis_files=basis_files,cross_edges=cross_edges,family=family,reduction_ratio=reduction_ratio,attention_dimension=attention_dimension,heads=heads,alignment_tokens=alignment_tokens,nested_rhos=nested_rhos,s_axis_permutation=s_axis_permutation,kernel_size=kernel_size,r=r,h=h,group_count=group_count,bottleneck_rank=bottleneck_rank)
    inserted=builder.steps[len(old):]
    builder.steps[:]=[row for i,row in enumerate(old) if i not in delayed]+inserted+[row for i,row in enumerate(old) if i in delayed]
    meta.update(shape_inference='representative_features',native_edges_rewired=False,
                deferred_native_edges=[old[i][0] for i in sorted(delayed)])
    return result,meta


class ParallelResidualSum(nn.Module):
    def forward(self, host_output, new_delta):
        return host_output + new_delta


def attach_parallel_to_nodes(builder, configs, samples, *, host_prefix="bridge_0_",
                             addition_prefix="bridge_1_", merge_name="bridge_parallel_merge"):
    """Keep the host's state keys and route both exchanges from pre-write Nodes."""
    host, addition = configs
    _, host_meta = attach_to_nodes(builder, host['nodes'], prefix=host_prefix, samples=samples,
                                   **{k:v for k,v in host.items() if k!='nodes'})
    host_steps = list(builder.steps)
    names = host['nodes']
    specs = [FeatureSpec(n, samples[n].shape[1], tuple(samples[n].shape[2:])) for n in names]
    inputs = {n:builder.by_name[n].id for n in names}
    _, new_meta = attach_group(builder.node, builder.edge, specs, inputs, addition_prefix,
                               **{k:v for k,v in addition.items() if k not in ('nodes','parallel_to')})
    modules={e.name:e.edge_operations[0].function for e in builder.edges}
    modules[addition_prefix+'exchange'].delta_only=True
    new_steps = builder.steps[len(host_steps):]
    packet=builder.node('bridge_parallel_output' if merge_name=='bridge_parallel_merge' else merge_name+'_output')
    builder.edge(merge_name,ParallelResidualSum(),
                 [host_meta['communication_node'],new_meta['communication_node']],[packet])
    merge_step=builder.steps[-1]
    exchange_id=next(e.id for e in builder.edges if e.name==host_prefix+'exchange')
    insertion=next(i for i,row in enumerate(host_steps) if row[0]==exchange_id)+1
    # The old return edges stay named identically and still write original Node IDs.
    reordered=host_steps[:insertion]+[new_steps[0],merge_step]+host_steps[insertion:]
    return_ids={e.id for e in builder.edges if e.name.startswith(host_prefix) and e.name.endswith('_return')}
    builder.steps[:]=[(eid,[packet] if eid in return_ids else heads,tails) for eid,heads,tails in reordered]
    for meta in (host_meta,new_meta):
        meta.update(topology='parallel_prewrite_residual',native_node_ids=inputs,
                    formula='X + delta_host(X) + delta_new(X)')
    return [host_meta,new_meta]
