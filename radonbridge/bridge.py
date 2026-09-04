"""Task-agnostic many-in/many-out R&B groups, expanded into explicit MHD edges."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn
from .projector import Projector, Return, orientations

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


def attach_group(node,edge,specs,inputs,prefix,upsilon=(1.,1.,.03125),mode="radon",kernel_size=3):
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
