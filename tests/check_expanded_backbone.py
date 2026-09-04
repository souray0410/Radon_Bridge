"""Official ResNet34, real batch4 gradient agreement and zero-bridge identity."""
import os,gc,json,hashlib
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from radonbridge.model import PilotGraph
from radonbridge.optimization import configure_optimizer,clip_task_gradients

def main():
 torch.set_num_threads(3);torch.manual_seed(721)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 protocol=json.load(open('experiments/008-expanded-validation/qualification_protocol.json'))
 recipe=protocol['recipes'][-1]
 g=PilotGraph(backbone='resnet34',device='cuda',cfp_size=224,loss_reduction='sum');g.graph.eval()
 assert len(g.modules_by_name()['cfp_stage1'][-1])==3 and len(g.modules_by_name()['cfp_stage4'])==3
 assert len(g.modules_by_name()['cfp_stage2'])==4 and len(g.modules_by_name()['cfp_stage3'])==6
 opt=configure_optimizer(g,recipe)
 c=torch.randn(4,2,3,224,224,device='cuda');o=torch.randn(4,2,1,32,96,96,device='cuda');y=torch.tensor([0,1,0,1],device='cuda')
 opt.zero_grad(set_to_none=True);_,loss=g.forward(c,o,y);g.backward()
 saved=[None if p.grad is None else p.grad.detach().cpu().clone() for p in g.graph.parameters()]
 opt.zero_grad(set_to_none=True);_,native=g.native_forward(c,o,y);native.backward();err=0.
 for p,a in zip(g.graph.parameters(),saved):
  assert (p.grad is None)==(a is None)
  if a is not None:
   b=p.grad.detach().cpu();err=max(err,float((a-b).abs().max()));assert torch.allclose(a,b,atol=2e-6,rtol=1e-4)
 assert torch.allclose(loss,native)
 clip_task_gradients(g);opt.step()
 with torch.no_grad():logits,_=g.forward(c,o,y);reference={k:v.cpu().clone() for k,v in logits.items()}
 state=g.save_state();peak=torch.cuda.max_memory_allocated()/1024**2
 del g,opt,saved,loss,native,logits;gc.collect();torch.cuda.empty_cache()
 bridge=PilotGraph('radon',backbone='resnet34',device='cuda',cfp_size=224,bridge_stages=(3,),upsilon=(1.,1.,1/32),loss_reduction='sum')
 bridge.load_state(state);bridge.graph.eval()
 with torch.no_grad():z,_=bridge.forward(c,o,y)
 assert all(torch.equal(reference[k],v.cpu()) for k,v in z.items())
 root=Path(os.environ['TORCH_HOME'])/'hub/checkpoints'
 weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('resnet*.pth')}
 print(json.dumps({'backbone':'resnet34','official_weight_files_sha256':weights,'blocks_checked':[3,4,6,3],'batch':4,'mhd_native_max_gradient_error':err,'zero_bridge_exact':True,'finite_optimizer_step':True,'peak_allocated_mib':peak}))
if __name__=='__main__':main()
