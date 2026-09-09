"""Explicit contract for a native MHD graph; no image/task names are assumed."""
from dataclasses import dataclass,field
from typing import Mapping
import torch
from .graph import MHDBuilder

@dataclass
class NativeNetwork:
    builder: MHDBuilder
    input_names: tuple
    prediction_nodes: Mapping[str,str]
    loss_node: str
    probe_inputs: Mapping[str,torch.Tensor]
    branches: tuple
    native_checkpoint_modules: tuple
    source_modules: Mapping[str,tuple]
    loss_scaler_edge: str | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.builder,MHDBuilder):raise TypeError('Import a native MHDBuilder with explicit Nodes and Edges')
        names=set(self.builder.by_name);modules={e.name for e in self.builder.edges}
        if len(self.input_names)!=len(set(self.input_names)) or set(self.probe_inputs)!=set(self.input_names):
            raise ValueError('Each input must have one declared probe tensor')
        if not set(self.input_names)<=names or not set(self.prediction_nodes.values())<=names or self.loss_node not in names:
            raise ValueError('Declared graph inputs/outputs/loss must exist')
        if not self.prediction_nodes or len(set(self.prediction_nodes.values()))!=len(self.prediction_nodes):raise ValueError('Distinct named predictions required')
        if set(self.source_modules)!=set(self.branches) or not set(self.native_checkpoint_modules)<=modules:
            raise ValueError('Explicit native checkpoint and source ownership required')
        owned=[n for group in self.source_modules.values() for n in group]
        if len(owned)!=len(set(owned)) or not set(owned)<=set(self.native_checkpoint_modules):raise ValueError('Source checkpoint ownership overlaps or is unknown')
        if self.loss_scaler_edge is not None and self.loss_scaler_edge not in modules:raise ValueError('Unknown loss scaler')
        if any(n.startswith('bridge_') for n in modules):raise ValueError('Native network reserves bridge_ for attached communication')
        return self
