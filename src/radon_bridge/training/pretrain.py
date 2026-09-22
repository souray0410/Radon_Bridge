"""Two independently pretrained task paths, intermediate R&B, per-task macro-F1."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from radon_bridge.data.dataset import PairedDataset
from radon_bridge.models.model import PilotGraph
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.training.optimization import configure_optimizer, clip_task_gradients


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    start=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError("Output must be empty")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise RuntimeError("Commit source first")
    protocol=json.loads(Path(args.protocol).read_text());seed=protocol["seed"]
    if protocol.get("loss_reduction") != "sum" or protocol.get("clip_policy") != "per_task":
        raise ValueError("New runs require explicit sum losses and per_task clipping; use archived source for experiment005")
    torch.set_num_threads(3);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    from mhd_models.scheduling.gpu_budget import configure_allocator
    configure_allocator(0)
    train=PairedDataset(args.data,"train",224);val=PairedDataset(args.data,"validation",224)
    assert not ({r["participant_id"] for r in train.rows}&{r["participant_id"] for r in val.rows})
    weight=Path(os.environ["TORCH_HOME"])/"hub/checkpoints/resnet18-f37072fd.pth"
    config=vars(args)|{"protocol":protocol,"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                      "mhd_commit":subprocess.check_output(["git","-C","third_party/MHD_Framework","rev-parse","HEAD"],text=True).strip(),
                      "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"pretraining":"ImageNet ResNet18; OCT inflated from 2D, not OCT pretraining",
                      "weight_sha256":hashlib.sha256(weight.read_bytes()).hexdigest(),
                      "audit":json.loads((Path(args.data)/"audit.json").read_text()),"cfp_audit":json.loads((Path(args.data)/"cfp224/audit.json").read_text())}
    (out/"config.json").write_text(json.dumps(config,indent=2))
    def budget():
        if time.monotonic()-start>protocol["max_minutes"]*60:raise RuntimeError("Wall-time budget exceeded")
    def memory():
        data=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],text=True)
        used=max([int(line.split(',')[1]) for line in data.splitlines() if line.split(',')[0].strip()==str(os.getpid())]or[0])
        if used>9728:raise RuntimeError("Own-process memory >9.5 GiB")
        return used
    def loader(dataset,epoch=None):
        return DataLoader(dataset,batch_size=4,shuffle=epoch is not None,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(seed+(epoch or 0)))
    def training(g):
        g.graph.train()
        for m in g.graph.modules():
            if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)):m.eval()
    @torch.no_grad()
    def evaluate(g,dataset,save=None):
        g.graph.eval();preds={k:[] for k in g.branches};ys=[];ids=[]
        for c,o,y,keys in loader(dataset):
            logits,_=g.forward(c.cuda(),o.cuda(),y.cuda())
            for k in preds:preds[k].append(logits[k].softmax(1).cpu().numpy())
            ys.extend(y.numpy());ids.extend(keys)
        arrays={k:np.concatenate(v) for k,v in preds.items()};y=np.asarray(ys)
        if save:np.savez(save,ids=np.asarray(ids),y=y,**arrays)
        result={k:classification_metrics(y,p) for k,p in arrays.items()}
        return {"tasks":result,"mean_task_macro_f1":float(np.mean([r["macro_f1"] for r in result.values()]))}
    peak=0
    def fit(g,name,epochs,warm=False):
        nonlocal peak
        opt=configure_optimizer(g,protocol,warm);best=evaluate(g,val);beststate=g.save_state();bestepoch=0
        (out/(name+"_model.json")).write_text(json.dumps({"groups":g.communication_groups,"parameters":sum(p.numel() for p in g.graph.parameters()),"trainable_parameters":sum(p.numel() for p in g.graph.parameters() if p.requires_grad)},indent=2))
        for epoch in range(epochs):
            budget();training(g);losses={k:0. for k in g.branches};n=0;t=time.monotonic()
            for c,o,y,_ in loader(train,epoch):
                opt.zero_grad(set_to_none=True);_,loss=g.forward(c.cuda(),o.cuda(),y.cuda());g.backward()
                clip_task_gradients(g,protocol.get("clip_max_norm",5.));opt.step()
                for k in losses:losses[k]+=float(g.by_name[k+"_loss"].feature_message.current_state.detach())*len(y)
                n+=len(y);budget()
            result=evaluate(g,val);peak=max(peak,memory())
            row={"arm":name,"epoch":epoch+1,"train_task_losses":{k:v/n for k,v in losses.items()},"validation":result,"seconds":time.monotonic()-t,"peak_process_mib":peak}
            with (out/"history.jsonl").open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({"arm":name,"epoch":epoch+1,"macro_f1":{k:v["macro_f1"] for k,v in result["tasks"].items()},"peak_process_mib":peak}),flush=True)
            if result["mean_task_macro_f1"]>best["mean_task_macro_f1"]:
                best=result;beststate=g.save_state();bestepoch=epoch+1
            torch.save({"model":g.save_state(),"optimizer":opt.state_dict(),"epoch":epoch+1,"config":config},out/(name+"_last.pt"))
        last=evaluate(g,val,out/(name+"_last_predictions.npz"));g.load_state(beststate)
        torch.save({"model":beststate,"epoch":bestepoch,"config":config},out/(name+"_best.pt"))
        return beststate,{"fixed_last":last,"selected":evaluate(g,val,out/(name+"_selected_predictions.npz")),"selected_epoch":bestepoch,"selected_train":evaluate(g,train)}
    warmstates={};warmresults={}
    for branch in ("cfp","oct"):
        g=PilotGraph(seed=seed,device="cuda",backbone="resnet18",cfp_size=224,modalities=branch,loss_reduction="sum")
        warmstates[branch],warmresults[branch]=fit(g,"warm_"+branch,protocol["warmup_epochs"],True)
        del g;torch.cuda.empty_cache()
    initial=warmstates["cfp"]|warmstates["oct"];del warmstates
    results={}
    for spec in protocol["arms"]:
        g=PilotGraph(spec["mode"],seed,"cuda",backbone="resnet18",cfp_size=224,bridge_stages=tuple(spec["stages"]),upsilon=tuple(protocol["upsilon"]),loss_reduction="sum")
        g.load_state(initial)
        _,results[spec["id"]]=fit(g,spec["id"],protocol["arm_epochs"])
        del g;torch.cuda.empty_cache()
    # Frozen recipe, paired participant bootstrap, fixed final epoch only.
    comparisons=[]
    with np.load(out/"independent_last_predictions.npz") as baseline:
        for spec in protocol["arms"][1:]:
            with np.load(out/(spec["id"]+"_last_predictions.npz")) as other:
                assert np.array_equal(baseline['ids'],other['ids']) and np.array_equal(baseline['y'],other['y'])
                for task in ("cfp","oct"):
                    a=other[task].argmax(1);b=baseline[task].argmax(1);y=baseline['y'];rng=np.random.default_rng(710);ds=[]
                    for _ in range(1000):
                        ix=rng.integers(0,len(y),len(y));ds.append(f1_score(y[ix],a[ix],labels=[0,1],average='macro',zero_division=0)-f1_score(y[ix],b[ix],labels=[0,1],average='macro',zero_division=0))
                    comparisons.append({"arm":spec["id"],"task":task,"delta_macro_f1":results[spec["id"]]["fixed_last"]["tasks"][task]["macro_f1"]-results["independent"]["fixed_last"]["tasks"][task]["macro_f1"],"paired_bootstrap_95pct":np.quantile(ds,[.025,.975]).tolist()})
    report={"warmup":warmresults,"arms":results,"comparisons":comparisons,"minutes":(time.monotonic()-start)/60,"peak_process_mib":peak,"test_used":False,"limitations":["one seed","256 train and 128 repeatedly inspected validation participants","exploratory intervals without multiplicity adjustment","warmup selected on validation macro-F1","no claim of effectiveness from this pilot alone"]}
    (out/"summary.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol',required=True);p.add_argument('--lock',required=True)
    args=p.parse_args()
    with open(args.lock,'a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args)
