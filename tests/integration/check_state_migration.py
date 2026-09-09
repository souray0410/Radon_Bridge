import sys,json,hashlib,importlib
from pathlib import Path
import torch
from torch import nn
mode,out=sys.argv[1],Path(sys.argv[2]);out.mkdir(exist_ok=True)
old=mode=='before'
bridge=importlib.import_module('radon_bridge.bridge' if old else 'radon_bridge.methods.operator')
MHDBuilder=importlib.import_module('radon_bridge.graph' if old else 'radon_bridge.models.graph').MHDBuilder
basis=importlib.import_module('radon_bridge.svd_basis' if old else 'radon_bridge.methods.basis')
def digest(values):
 h=hashlib.sha256()
 for k,v in sorted(values.items()):
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class Stage(nn.Module):
 def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.1,dtype=torch.float64))
 def forward(self,x):return torch.tanh(self.weight*x)
class Loss(nn.Module):
 def forward(self,*xs):return sum(x.square().mean() for x in xs)
results=[];torch.set_num_threads(2)
for compression in ['learned_projected','fixed_svd_channel']:
 torch.manual_seed(713)
 b=MHDBuilder();inputs={};cuts=[];outputs=[];refs={}
 for i,shape in enumerate([(3,4),(3,4,3)]):
  key=f'source{i}';node=b.node(key)
  inputs[key]=torch.randn(2,4,*shape,dtype=torch.float64,requires_grad=True)
  cut=b.node(key+'_stage');b.edge(key+'_stage',Stage(),[node],[cut]);cuts.append(key+'_stage')
  tail=b.node(key+'_tail');b.edge(key+'_tail',Stage(),[cut],[tail]);outputs.append(tail)
  f=torch.randn(4,12,dtype=torch.float64,generator=torch.Generator().manual_seed(100+i))
  basis_dir=out/compression/mode;basis_dir.mkdir(parents=True,exist_ok=True)
  refs[cuts[-1]]=basis.save_basis(f@f.T,12,basis_dir,cuts[-1],3416,{'synthetic_test':True})
 loss=b.node('loss');b.edge('objective',Loss(),outputs,[loss]);samples=b.native_forward(inputs)
 kw={'basis_files':refs} if compression=='fixed_svd_channel' else {}
 bridge.attach_to_nodes(b,cuts,prefix='bridge_',M=4,S=11,rho=.5,samples=samples,compression=compression,**kw)
 g=b.compile('cpu').double()
 for module in g.modules():
  if isinstance(module,bridge.LinearMixer):nn.init.normal_(module.conv.weight,std=.02)
 state=out/(compression+'.pt')
 if old:torch.save(g.state_dict(),state)
 else:g.load_state_dict(torch.load(state,weights_only=True),strict=True)
 b.set_inputs(inputs);g.forward(levels=b.forward_levels);g.backward(levels=b.backward_levels)
 row={'state':digest(g.state_dict()),'output':b.by_name['loss'].feature_message.current_state.detach().tolist(),'gradients':digest({k:p.grad for k,p in g.named_parameters() if p.grad is not None}),'input_gradients':digest({k:x.grad for k,x in inputs.items()}),'nodes':sorted((n.id,n.name) for n in g.nodes)}
 torch.optim.SGD(g.parameters(),lr=1e-5).step();row['updated_state']=digest(g.state_dict());results.append(row)
(out/(mode+'.json')).write_text(json.dumps(results,indent=2))
if not old:assert json.loads(json.dumps(results))==json.loads((out/'before.json').read_text());print('PASS learned and fixed-SVD: strict state_dict, outputs, parameter/input gradients, Node IDs and update')
