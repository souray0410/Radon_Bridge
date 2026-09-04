"""Read-only checkpoint diagnostics. No new fitting or participant export."""
import fcntl,json,os,gc
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from radonbridge.model import PilotGraph
from radonbridge.metrics import classification_metrics
from radonbridge.bridge import LinearMixer

def main():
 os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
 torch.set_num_threads(3);torch.manual_seed(781)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
 from radonbridge.data import PairedDataset
 root=Path('/data/mengh/RadonBridge/runs/exp005');out=Path('/data/mengh/RadonBridge/runs/exp006_audit');out.mkdir(exist_ok=True)
 train=PairedDataset('/data/mengh/RadonBridge/cache/pilot256_128','train',224)
 val=PairedDataset('/data/mengh/RadonBridge/cache/pilot256_128','validation',224)
 summary=json.loads((root/'summary.json').read_text());report={}
 for name,mode,stages in [('independent','baseline',()),('radon_s3','radon',(3,)),('radon_s2_s3','radon',(2,3))]:
  g=PilotGraph(mode,device='cuda',backbone='resnet18',cfp_size=224,bridge_stages=stages,upsilon=(1.,1.,.03125))
  ckpt=torch.load(root/(name+'_last.pt'),map_location='cpu',weights_only=False);g.load_state(ckpt['model']);del ckpt
  g.graph.eval();result={};store={}
  with torch.no_grad():
   for split,data in [('train',train),('validation',val)]:
    probs={k:[] for k in g.branches};ys=[];delta_stats={}
    for c,o,y,_ in DataLoader(data,batch_size=4,shuffle=False):
     logits,_=g.forward(c.cuda(),o.cuda(),y.cuda());ys.extend(y.numpy())
     for k in probs:probs[k].append(logits[k].softmax(1).cpu().numpy())
     for stage in stages:
      for k in g.branches:
       native=g.by_name[f'{k}_stage{stage}'].feature_message.current_state
       delta=g.by_name[f'bridge_s{stage}_{k}_delta'].feature_message.current_state
       key=f's{stage}_{k}';acc=delta_stats.setdefault(key,[0.,0.]);acc[0]+=float(delta.square().sum());acc[1]+=float(native.square().sum())
    arrays={k:np.concatenate(v) for k,v in probs.items()};store[split]=(np.array(ys),arrays)
    result[split]={k:classification_metrics(ys,p) for k,p in arrays.items()}
    result[split+'_delta_over_feature_l2']={k:(a/b)**.5 for k,(a,b) in delta_stats.items()}
    if split=='validation':
     with np.load(root/(name+'_last_predictions.npz')) as saved:
      assert all(np.allclose(saved[k],arrays[k],atol=1e-6) for k in arrays)
   # Direct held-out intervention, only first deterministic four participants.
   c,o,y,_=next(iter(DataLoader(val,batch_size=4,shuffle=False)));c,o,y=c.cuda(),o.cuda(),y.cuda()
   pred,_=g.forward(c,o,y);pred={k:v.clone() for k,v in pred.items()}
   shuffled_oct,_=g.forward(c,o.roll(1,0),y);cross_cfp=float((shuffled_oct['cfp']-pred['cfp']).norm())
   shuffled_cfp,_=g.forward(c.roll(1,0),o,y);cross_oct=float((shuffled_cfp['oct']-pred['oct']).norm())
   mixers=[m for m in g.modules_by_name().values() if isinstance(m,LinearMixer)]
   weights=[m.conv.weight.detach().clone() for m in mixers]
   for m in mixers:m.conv.weight.zero_()
   disabled,_=g.forward(c,o,y)
   result['four_participant_intervention']={'shuffled_other_logit_l2':{'cfp':cross_cfp,'oct':cross_oct},'all_bridges_disabled_logit_l2':{k:float((disabled[k]-pred[k]).norm()) for k in pred}}
   for m,w in zip(mixers,weights):m.conv.weight.copy_(w)
  # Compare two consecutive real-data forwards/backwards with same frozen stage policy.
  for key,module in g.modules_by_name().items():
   enabled=key.endswith('_head') or key.startswith('bridge_') or key.endswith('_stage3') or key.endswith('_stage4')
   for p in module.parameters():p.requires_grad_(enabled)
  checks=[]
  for step,(c,o,y,_) in enumerate(DataLoader(train,batch_size=2,shuffle=False)):
   if step==2:break
   c,o,y=c.cuda(),o.cuda(),y.cuda();g.graph.zero_grad(set_to_none=True)
   _,loss=g.forward(c,o,y);g.backward()
   grads={key:[None if p.grad is None else p.grad.detach().cpu().clone() for p in module.parameters()] for key,module in g.modules_by_name().items()}
   norms={key:sum(float(x.square().sum()) for x in values if x is not None)**.5 for key,values in grads.items()}
   total=sum(x*x for x in norms.values())**.5
   g.graph.zero_grad(set_to_none=True);_,native=g.native_forward(c,o,y);native.backward();maxerr=0.;mismatches=[]
   for key,module in g.modules_by_name().items():
    for p,a in zip(module.parameters(),grads[key]):
     assert (p.grad is None)==(a is None)
     if a is not None:
      b=p.grad.detach().cpu();err=float((a-b).abs().max());maxerr=max(maxerr,err)
      if not torch.allclose(a,b,atol=2e-6,rtol=1e-4):mismatches.append({"module":key,"max_abs":err,"relative_l2":float((a-b).norm()/(a.norm()+1e-20)),"shape":list(a.shape)})
   checks.append({'step':step,'loss':float(loss.detach()),'native_loss':float(native.detach()),'max_gradient_error':maxerr,'mismatches':mismatches,'module_gradient_norms':norms,'global_gradient_norm':total,'global_clip_multiplier_at_5':min(1.,5/(total+1e-6))})
  result['real_data_repeated_gradient_checks']=checks;report[name]=result
  (out/'report.json').write_text(json.dumps(report,indent=2))
  del g,grads,loss,native,weights,mixers;gc.collect();torch.cuda.empty_cache()
  print(json.dumps({'finished':name,'train_f1':{k:v['macro_f1'] for k,v in result['train'].items()},'delta_ratio':result['validation_delta_over_feature_l2'],'intervention':result['four_participant_intervention']}),flush=True)
 report['notes']=['Read-only last checkpoints; no fitting','Real gradient check uses eval BN and actual trainable stages','Intervention uses four validation participants and is diagnostic only','Global clipping on concatenated task gradients couples updates even without feature communication']
 (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
 with open('/data/mengh/RadonBridge/.active.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
