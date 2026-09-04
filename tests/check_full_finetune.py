"""Full-size batch4 acceptance: all stages update, MHD agreement and cross-task early gradients."""
import os,time,json,gc,argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from radonbridge.model import PilotGraph,SliceMaxPool3d
from radonbridge.optimization import configure_optimizer,clip_task_gradients

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--handoff-ratio",type=float,default=1/32);args=parser.parse_args()
 torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 pool=SliceMaxPool3d(torch.nn.MaxPool2d(3,2,1)); reference=torch.nn.MaxPool3d((1,3,3),(1,2,2),(0,1,1))
 for ties in [False,True]:
  x=torch.randn(2,3,4,11,13,dtype=torch.double).relu() if ties else torch.randn(2,3,4,11,13,dtype=torch.double)
  x.requires_grad_();a=pool(x);b=reference(x);assert torch.equal(a,b)
  grad=torch.randn_like(a);ga=torch.autograd.grad(a,x,grad)[0];gb=torch.autograd.grad(b,x,grad)[0];assert torch.equal(ga,gb)
 recipe={'adapt_stages':[1,2,3,4],'training_regime':'full_finetune','backbone_lr':3e-6,'head_bridge_lr':1e-4,'weight_decay':.01}
 g=PilotGraph('radon',backbone='resnet18',device='cuda',cfp_size=224,bridge_stages=(3,),upsilon=(1.,1.,args.handoff_ratio),loss_reduction='sum')
 opt=configure_optimizer(g,recipe);assert all(p.requires_grad for p in g.graph.parameters())
 modules={k:m for k,m in g.modules_by_name().items() if list(m.parameters())}
 before={k:[p.detach().cpu().clone() for p in m.parameters()] for k,m in modules.items()}
 bn=[m for m in g.graph.modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))];counts=[int(m.num_batches_tracked) for m in bn]
 c=torch.randn(4,2,3,224,224,device='cuda');o=torch.randn(4,2,1,32,96,96,device='cuda');y=torch.tensor([0,1,0,1],device='cuda')
 times=[]
 for step in range(5):
  g.graph.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
  g.forward(c,o,y);g.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.graph.parameters())
  clip_task_gradients(g);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
 updates={k:max(float((p.detach().cpu()-old).abs().max()) for p,old in zip(m.parameters(),before[k])) for k,m in modules.items()}
 assert all(v>0 for v in updates.values()),updates
 assert all(int(m.num_batches_tracked)>n for m,n in zip(bn,counts))
 g.graph.eval();opt.zero_grad(set_to_none=True);_,loss=g.forward(c,o,y);g.backward()
 saved=[p.grad.detach().cpu().clone() for p in g.graph.parameters()]
 opt.zero_grad(set_to_none=True);logits,native=g.native_forward(c,o,y);native.backward();err=0.
 for p,a in zip(g.graph.parameters(),saved):
  b=p.grad.detach().cpu();err=max(err,float((a-b).abs().max()));assert torch.allclose(a,b,atol=2e-6,rtol=1e-4)
 del saved,logits,loss,native;gc.collect()
 opt.zero_grad(set_to_none=True);logits,_=g.native_forward(c,o,y);cross={}
 for target,source in [('cfp','oct'),('oct','cfp')]:
  weight=next(modules[source+'_stage1'].parameters())
  gradient=torch.autograd.grad(torch.nn.functional.cross_entropy(logits[target],y),weight,retain_graph=True)[0]
  cross[target+'_loss_to_'+source+'_stem']=float(gradient.norm());assert cross[target+'_loss_to_'+source+'_stem']>0
 total=sum(p.numel() for p in g.graph.parameters())
 print(json.dumps({'upsilon_H':args.handoff_ratio,'depth_one_pool_forward_and_gradient_exact':True,'batch_size':4,'all_parameters_trainable':True,'total_parameters':total,'trainable_parameters':total,'module_max_parameter_updates':updates,'batchnorm_statistics_updated':True,'mhd_native_max_gradient_error':err,'cross_task_early_gradient_norms_after_learning':cross,'step_seconds':times,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}))
if __name__=='__main__':main()
