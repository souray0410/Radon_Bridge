"""Minimal builder for the pinned MHD V4 node/edge topology, not another backend."""
import torch
from V4.MHD_Framework_V4 import MHD_Node,MHD_Edge,MHD_Topo,MHD_Graph

class MHDBuilder:
    def __init__(self):self.nodes=[];self.edges=[];self.steps=[];self.by_name={}
    def node(self,name):
        if name in self.by_name:raise ValueError(name)
        n=MHD_Node(len(self.nodes),name,MHD_Node.Message(torch.zeros(1)),aggregation="replace")
        self.nodes.append(n);self.by_name[name]=n;return n.id
    def edge(self,name,fn,heads,tails):
        eid=len(self.edges);self.edges.append(MHD_Edge(eid,name,[MHD_Edge.Operation(fn)]));self.steps.append((eid,heads,tails))
    def compile(self,device="cpu"):
        roles=[];sorts=[]
        for eid,heads,tails in self.steps:
            r=torch.zeros(len(self.edges),len(self.nodes),dtype=torch.long);s=torch.zeros_like(r)
            for order,n in enumerate(heads):r[eid,n]=-1;s[eid,n]=order
            for order,n in enumerate(tails,len(heads)):r[eid,n]=1;s[eid,n]=order
            roles.append(r);sorts.append(s)
        self.forward_levels=list(range(len(roles)));self.backward_levels=list(range(len(roles),2*len(roles)))
        return MHD_Graph(set(self.nodes),set(self.edges),{MHD_Topo(roles+[-r for r in reversed(roles)],sorts+list(reversed(sorts)))},device=torch.device(device))
    def set_inputs(self,values):
        for k,v in values.items():
            self.by_name[k].feature_message.initial_state=v;self.by_name[k].feature_message.current_state=v
    def native_forward(self,values):
        values={self.by_name[k].id:v for k,v in values.items()}
        for eid,heads,tails in self.steps:
            result=self.edges[eid].edge_operations[0].function(*[values[n] for n in heads])
            for n,v in zip(tails,result if isinstance(result,tuple) else (result,)):values[n]=v
        return {k:values[n.id] for k,n in self.by_name.items() if n.id in values}
