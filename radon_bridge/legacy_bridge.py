"""Task-agnostic many-in/many-out R&B groups, expanded into explicit MHD edges."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn
from .legacy_projector import Projector, Return, orientations

@dataclass(frozen=True)
class FeatureSpec:
    key: str
    channels: int
    shape: tuple
    mesh_reference: tuple
    span_reference: int | None = None
    coordinate_mode: str = "normalized"
    spacing: tuple | None = None

    def geometry(self):
        if self.channels<1 or not self.shape or min(self.shape)<2:raise ValueError("Invalid native feature")
        if len(self.mesh_reference)!=len(self.shape)-1 or any(m<1 for m in self.mesh_reference):raise ValueError("Invalid reference mesh")
        if self.coordinate_mode=="normalized":spacing=(2/max(self.shape),)*len(self.shape)
        elif self.coordinate_mode=="index":spacing=(1.,)*len(self.shape)
        elif self.coordinate_mode=="physical" and self.spacing is not None:spacing=tuple(self.spacing)
        else:raise ValueError("Explicit physical spacing required")
        if len(spacing)!=len(self.shape) or min(spacing)<=0:raise ValueError("Invalid spacing")
        radius=float(np.linalg.norm((np.asarray(self.shape)+1)*np.asarray(spacing)/2))
        span=self.span_reference if self.span_reference is not None else math.ceil(2*radius/min(spacing))+1
        if span<1:raise ValueError("Invalid reference span")
        return spacing,radius,span

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

class Add(nn.Module):
    def forward(self,a,b):return a+b


def attach_group_expanded(node,edge,specs,inputs,prefix,upsilon=(1.,1.,.03125),mode="radon",kernel_size=3):
    """Only sees feature nodes; returns the SAME participant keys, never task heads."""
    keys=[s.key for s in specs]
    if len(set(keys))!=len(keys) or set(keys)!=set(inputs) or not keys:raise ValueError("Participant identity mismatch")
    if mode not in ("off","radon","scrambled","self","random"):raise ValueError(mode)
    if len(upsilon)!=3 or any(not 0<u<=1 for u in upsilon):raise ValueError("Use explicit off; upsilon in (0,1]")
    geometries=[s.geometry() for s in specs];reference_span=max(g[2] for g in geometries)
    span=max(1,round(upsilon[1]*reference_span));meta={"mode":mode,"upsilon":list(upsilon),"span_reference":reference_span,"span":span,"participants":[]}
    if mode=="off":return dict(inputs),meta
    projected={};projectors={};widths={}
    for spec,(spacing,radius,s0) in zip(specs,geometries):
        key=spec.key;mesh=tuple(max(1,round(upsilon[0]*m)) for m in spec.mesh_reference)
        p=Projector(spec.shape,mesh,span,scramble=mode=="scrambled",spacing=spacing,random_projection=mode=="random")
        width=spec.channels*math.prod(mesh);h=max(1,round(upsilon[2]*width));widths[key]=(width,h);projectors[key]=p
        n=node(prefix+key+"_projected");edge(prefix+key+"_project",p,[inputs[key]],[n])
        u=node(prefix+key+"_handoff");edge(prefix+key+"_compress",nn.Conv1d(width,h,1,bias=False),[n],[u]);projected[key]=u
        ns,_=orientations(mesh)
        meta["participants"].append(asdict(spec)|{"mesh":mesh,"span_reference":s0,"support":[-radius,radius],"effective_spacing":spacing,"P":width,"H":h,"unique_directions":len(np.unique(np.round(ns,10),axis=0)),"operator_scale":p.scale,"projection_kind":p.projection_kind,"projection_random_seed":p.random_seed})
    mixed=[node(prefix+k+"_mixed") for k in keys]
    edge(prefix+"mixer",LinearMixer([widths[k][1] for k in keys],kernel_size,mode=="self"),[projected[k] for k in keys],mixed)
    outputs={}
    for key,source in zip(keys,mixed):
        width,h=widths[key];expanded=node(prefix+key+"_expanded")
        edge(prefix+key+"_expand",nn.Conv1d(h,width,1,bias=False),[source],[expanded])
        delta=node(prefix+key+"_delta");edge(prefix+key+"_return",Return(projectors[key]),[expanded],[delta])
        updated=node(prefix+key+"_updated");edge(prefix+key+"_residual",Add(),[inputs[key],delta],[updated]);outputs[key]=updated
    return outputs,meta


class BridgeExchange(nn.Module):
    """One communication hyperedge; native features are returned in a packed Tensor.

    Packing is required by MHD's Tensor Message contract. Return edges only
    unpack each participant's updated feature; they do not create native nodes.
    """
    def __init__(self, specs, upsilon, mode, kernel_size):
        super().__init__()
        # Reuse the exact existing numerical modules and their initialization
        # order, but keep their intermediate values internal to this operation.
        internal_nodes = []
        internal_edges = []
        def node(name):
            internal_nodes.append(name)
            return len(internal_nodes)-1
        def edge(name, fn, heads, tails):
            internal_edges.append((name, fn, heads, tails))
        initial = {s.key: node(s.key) for s in specs}
        outputs, self.metadata = attach_group_expanded(node, edge, specs, initial, '', upsilon, mode, kernel_size)
        self.operations = nn.ModuleList([row[1] for row in internal_edges])
        self.routes = [(row[2], row[3]) for row in internal_edges]
        self.operation_names = [row[0] for row in internal_edges]
        self.initial_ids = [initial[s.key] for s in specs]
        self.output_ids = [outputs[s.key] for s in specs]
        self.delta_ids = [internal_nodes.index(s.key+'_delta') for s in specs]
        self.keys = [s.key for s in specs]
        self.shapes = [(s.channels, *s.shape) for s in specs]
        self.lengths = [math.prod(shape) for shape in self.shapes]
        self.latest_inputs = self.latest_deltas = None

    def forward(self, *features):
        if len(features) != len(self.shapes):
            raise ValueError('Participant count changed')
        if any(tuple(x.shape[1:]) != shape for x, shape in zip(features, self.shapes)):
            raise ValueError('Native feature shape changed')
        if len({x.shape[0] for x in features}) != 1:
            raise ValueError('Participants must share batch alignment')
        values = dict(zip(self.initial_ids, features))
        for operation, (heads, tails) in zip(self.operations, self.routes):
            result = operation(*[values[n] for n in heads])
            for n, value in zip(tails, result if isinstance(result, tuple) else (result,)):
                values[n] = value
        self.latest_inputs = features
        self.latest_deltas = tuple(values[n] for n in self.delta_ids)
        return torch.cat([values[n].flatten(1) for n in self.output_ids], dim=1)


class ReturnParticipant(nn.Module):
    """Parameter-free routing back to an existing native Node."""
    def __init__(self, start, length, shape):
        super().__init__(); self.start=start; self.length=length; self.shape=tuple(shape)
    def forward(self, packet):
        return packet[:, self.start:self.start+self.length].reshape(packet.shape[0], *self.shape)


def attach_group(node, edge, specs, inputs, prefix, upsilon=(1.,1.,.03125), mode='radon', kernel_size=3):
    """Read existing Nodes, communicate once, write back at the next level."""
    if mode == 'off':
        return attach_group_expanded(node, edge, specs, inputs, prefix, upsilon, mode, kernel_size)
    if len({s.key for s in specs}) != len(specs) or set(inputs) != {s.key for s in specs}:
        raise ValueError('Participant identity mismatch')
    exchange = BridgeExchange(specs, upsilon, mode, kernel_size)
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


def attach_to_nodes(builder, node_names, *, prefix, samples=None, mesh_reference=None,
                    angular_resolution=4, upsilon=(1.,1.,.125), mode='radon', kernel_size=3):
    """Attach to a causal cut of existing networks without changing native edges.

    Infer C and spatial dimension from a representative forward pass. All other
    native operations are kept; consumers of selected feature versions are
    scheduled after the return level. Causally nested selections are rejected
    rather than silently running a downstream network on stale features.
    """
    if len(set(node_names)) != len(node_names) or not node_names:
        raise ValueError('Select distinct existing Nodes')
    specs=[];inputs={}
    for name in node_names:
        native=builder.by_name[name]
        feature=samples[name] if samples is not None else native.feature_message.current_state
        if not isinstance(feature,torch.Tensor) or feature.ndim<3:
            raise ValueError('Supply a representative [batch,channels,*spatial] feature for every participant')
        d=feature.ndim-2
        mesh=(mesh_reference or {}).get(name,(angular_resolution,)*(d-1))
        specs.append(FeatureSpec(name,feature.shape[1],tuple(feature.shape[2:]),tuple(mesh)))
        inputs[name]=native.id
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
    result,meta=attach_group(builder.node,builder.edge,specs,inputs,prefix,upsilon,mode,kernel_size)
    inserted=builder.steps[len(old):]
    builder.steps[:]=[row for i,row in enumerate(old) if i not in delayed]+inserted+[row for i,row in enumerate(old) if i in delayed]
    meta.update(shape_inference='representative_features',native_edges_rewired=False,
                deferred_native_edges=[old[i][0] for i in sorted(delayed)])
    return result,meta
