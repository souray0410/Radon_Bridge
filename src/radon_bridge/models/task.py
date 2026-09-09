"""Reusable MHD V4 execution/checkpoint layer, independent of network architecture.

This delegates to the existing MHD graph. It is not a second autograd engine.
"""
from radon_bridge.methods.integration import attach_communications

class MHDTaskGraph:
    def __init__(self,native,*,bridge_configs=(),device='cpu'):
        self.definition=native.validate();self.builder=native.builder
        self.nodes,self.edges,self.steps,self.by_name=(self.builder.nodes,self.builder.edges,self.builder.steps,self.builder.by_name)
        self.branches=tuple(native.branches);self.output_names=tuple(native.prediction_nodes)
        self.head_mode=native.metadata.get('head_mode','separate');self.task_fusion=native.metadata.get('task_fusion')
        self.communication_groups,self.separate_communication_clipping=attach_communications(native,bridge_configs)
        self.graph=self.builder.compile(device).float()
        self.forward_levels,self.backward_levels=self.builder.forward_levels,self.builder.backward_levels

    def forward_inputs(self,values,loss_scale=1.):
        if set(values)!=set(self.definition.input_names):raise ValueError('Provide exactly the declared native inputs')
        scaler=self.definition.loss_scaler_edge
        if scaler is not None:self.modules_by_name()[scaler].scale=loss_scale
        elif loss_scale!=1.:raise ValueError('This native graph has no loss-scaling edge')
        self.builder.set_inputs(values);self.graph.forward(levels=self.forward_levels)
        return {key:self.by_name[node].feature_message.current_state for key,node in self.definition.prediction_nodes.items()},self.by_name[self.definition.loss_node].feature_message.current_state

    def backward(self):self.graph.backward(levels=self.backward_levels)

    def native_forward_inputs(self,values):
        if set(values)!=set(self.definition.input_names):raise ValueError('Provide exactly the declared native inputs')
        result=self.builder.native_forward(values)
        return {key:result[node] for key,node in self.definition.prediction_nodes.items()},result[self.definition.loss_node]

    def modules_by_name(self):return {e.name:e.edge_operations[0].function for e in self.edges}

    def save_state(self):
        return {k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in self.modules_by_name().items()}

    def load_native_state(self,state,branch=None):
        expected=set(self.definition.native_checkpoint_modules if branch is None else self.definition.source_modules[branch])
        if set(state)!=expected:raise ValueError(f'Checkpoint native module mismatch: missing={expected-set(state)}, extra={set(state)-expected}')
        modules=self.modules_by_name()
        for key,values in state.items():modules[key].load_state_dict(values,strict=True)

    def load_complete_state(self,state,*,allow_new_bridge=False):
        modules=self.modules_by_name()
        allowed={k for k in modules if k.startswith('bridge_1_') or k=='bridge_parallel_merge'} if allow_new_bridge else set()
        if set(state)-set(modules) or set(modules)-set(state)-allowed:raise ValueError('Full checkpoint module names do not match')
        for key,values in state.items():modules[key].load_state_dict(values,strict=True)
