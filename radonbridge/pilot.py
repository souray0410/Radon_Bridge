"""Small, paired-control pilot; all reported outcomes are validation outcomes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss
from torch.utils.data import DataLoader
from .data import PairedDataset
from .model import PilotGraph


def atomic_json(path, value):
    tmp=path.with_suffix(".partial");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def metrics(y,p):
    return {"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),
            "accuracy_at_0.5":float(accuracy_score(y,p>=.5)),"log_loss":float(log_loss(y,p)),
            "brier":float(np.mean((y-p)**2))}


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    start=time.time(); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError("Run output must be empty; never append to an interrupted or completed run")
    torch.set_num_threads(3);torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device("cuda:0")
    total=torch.cuda.get_device_properties(device).total_memory
    # Allocator cap leaves room for CUDA context, libraries and transient use.
    torch.cuda.set_per_process_memory_fraction((8*1024**3)/total,device)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    train=PairedDataset(args.data,"train",args.cfp_size);val=PairedDataset(args.data,"validation",args.cfp_size)
    if set(r["participant_id"] for r in train.rows)&set(r["participant_id"] for r in val.rows):raise RuntimeError("Leakage")
    config=vars(args)|{"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
                      "train_participants":len(train),"validation_participants":len(val),
                      "pretraining":"torchvision ResNet18 ImageNet1K V1; OCT uses depth-averaged 3D inflation",
                      "trainable":"warmup: stage4 and classifier; arms: classifier and bridge only; BatchNorm statistics frozen",
                      "allocator_cap_gib":8,"process_stop_mib":9728,
                      "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                      "mhd_commit":subprocess.check_output(["git","-C","third_party/MHD_Project","rev-parse","HEAD"],text=True).strip()}
    config["source_branch"]=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip(): raise RuntimeError("Commit source before training")
    if args.cfp_size != 96:
        config["cfp_audit"]=json.loads((Path(args.data)/f"cfp{args.cfp_size}"/"audit.json").read_text())
    config["data_audit"]=json.loads((Path(args.data)/"audit.json").read_text())
    weight_file=Path(os.environ.get("TORCH_HOME",str(Path.home()/".cache/torch")))/"hub/checkpoints/resnet18-f37072fd.pth"
    if weight_file.exists():config["pretrained_sha256"]=hashlib.sha256(weight_file.read_bytes()).hexdigest()
    atomic_json(out/"config.json",config)
    def process_memory():
        r=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],text=True)
        used=max([int(s.split(",")[1]) for s in r.splitlines() if s.split(",")[0].strip()==str(os.getpid())] or [0])
        if used>9728:raise RuntimeError("Own process exceeded 9.5 GiB safety threshold")
        return used
    def check_budget():
        if time.time()-start>args.max_minutes*60:raise RuntimeError("Pilot time budget reached; checkpoints retained")
    def configure(g, phase="warmup"):
        for name,module in g.modules_by_name().items():
            enabled=name=="classifier" or any(s in name for s in ("compress","expand","mixer"))
            if phase=="warmup":enabled=name in ("cfp_stage4","oct_stage4","classifier")
            for p in module.parameters():p.requires_grad_(enabled)
        g.graph.train()
        for module in g.modules_by_name().values():
            for m in module.modules():
                if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)):m.eval()
        return torch.optim.AdamW([p for p in g.graph.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    def loader(dataset,epoch=None):
        return DataLoader(dataset,batch_size=args.batch,shuffle=epoch is not None,num_workers=2,
                          generator=torch.Generator().manual_seed(args.seed+(epoch or 0)),pin_memory=True)
    @torch.no_grad()
    def evaluate(g,dataset,save=None,ablate=None):
        g.graph.eval();ys=[];ps=[];ids=[]
        for c,o,y,keys in loader(dataset):
            c,o,y=c.to(device),o.to(device),y.to(device)
            if ablate=="cfp":c=torch.zeros_like(c)
            if ablate=="oct":o=torch.zeros_like(o)
            logits,_=g.forward(c,o,y)
            ps.extend(logits.softmax(1)[:,1].cpu().numpy());ys.extend(y.cpu().numpy());ids.extend(keys)
        y,p=np.asarray(ys),np.asarray(ps)
        if save is not None:np.savez(save,ids=np.asarray(ids),y=y,p=p)
        return metrics(y,p)
    history=[];peak_process=0
    def run(g,name,epochs,initial=None,phase="arms"):
        nonlocal peak_process
        if initial:g.load_state(initial)
        opt=configure(g,phase)
        # Include the common epoch-zero checkpoint in every continuation.
        # Also preserve fixed-last-epoch metrics, independent of validation selection.
        best=evaluate(g,val);best_state=g.save_state();best_epoch=0
        torch.save({"model":best_state,"epoch":0,"arm":name,"config":config,"validation":best},out/(name+"_best.pt"))
        for epoch in range(epochs):
            check_budget(); t=time.time();configure_training(g)
            loss_sum=0.;count=0
            for c,o,y,_ in loader(train,epoch):
                c,o,y=c.to(device),o.to(device),y.to(device)
                opt.zero_grad(set_to_none=True)
                _,loss=g.forward(c,o,y);g.backward()
                torch.nn.utils.clip_grad_norm_([p for p in g.graph.parameters() if p.requires_grad],5.)
                opt.step();loss_sum+=float(loss.detach())*len(y);count+=len(y)
                check_budget()
            vm=evaluate(g,val);peak_process=max(peak_process,process_memory())
            row={"arm":name,"epoch":epoch+1,"train_loss":loss_sum/count,**vm,
                 "seconds":time.time()-t,"process_mib":peak_process,
                 "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
                 "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2}
            history.append(row)
            with (out/"history.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            if best is None or vm["auroc"]>best["auroc"]:
                best=vm;best_state=g.save_state();best_epoch=epoch+1
                torch.save({"model":best_state,"optimizer":opt.state_dict(),"epoch":epoch+1,
                            "arm":name,"config":config,"validation":vm},out/(name+"_best.pt"))
            torch.save({"model":g.save_state(),"optimizer":opt.state_dict(),"epoch":epoch+1,
                        "arm":name,"config":config,"validation":vm},out/(name+"_last.pt"))
        last=evaluate(g,val,out/(name+"_last_predictions.npz"))
        g.load_state(best_state)
        actual=evaluate(g,val,out/(name+"_predictions.npz"))
        return best_state,actual|{"best_epoch":best_epoch,"fixed_last":last,"train":evaluate(g,train)}
    def configure_training(g):
        g.graph.train()
        for module in g.modules_by_name().values():
            for m in module.modules():
                if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)):m.eval()
    baseline=PilotGraph("baseline",args.seed,device,backbone="resnet18",cfp_size=args.cfp_size,modalities=args.modalities)
    # Verify real pretrained graph shapes and peak memory on one full batch.
    c,o,y,_=next(iter(loader(train)))
    opt=configure(baseline,"warmup");t=time.time()
    initial=baseline.save_state()
    _,loss=baseline.forward(c.to(device),o.to(device),y.to(device));baseline.backward();opt.step()
    torch.cuda.synchronize()
    print(json.dumps({"smoke_step_seconds":time.time()-t,"process_mib":process_memory(),
                      "initial_loss":float(loss.detach())}),flush=True)
    # Independent optimization diagnostic on sixteen training participants.
    # Restore the original parameters and discard optimizer state afterwards.
    tiny=[]
    for batch in loader(train,0):
        tiny.append(tuple(v.to(device) for v in batch[:3]))
        if len(tiny)==4:break
    losses=[];t=time.time()
    for step in range(40):
        c,o,y=tiny[step%len(tiny)]
        opt.zero_grad(set_to_none=True)
        z,l=baseline.forward(c,o,y);baseline.backward();opt.step()
        losses.append(float(l.detach()))
    with torch.no_grad():
        correct=0;total_n=0
        for c,o,y in tiny:
            z,_=baseline.forward(c,o,y);correct+=int((z.argmax(1)==y).sum());total_n+=len(y)
    overfit={"participants":total_n,"steps":40,"first4_loss":float(np.mean(losses[:4])),
             "last4_loss":float(np.mean(losses[-4:])),"accuracy":correct/total_n,"seconds":time.time()-t}
    atomic_json(out/"tiny_overfit.json",overfit);print(json.dumps({"tiny_overfit":overfit}),flush=True)
    del tiny,z,l
    baseline.load_state(initial);del initial,opt
    state,warm=run(baseline,"warmup",args.warmup_epochs,phase="warmup")
    # Occlusion is an input-use diagnostic, not a separately trained unimodal baseline.
    ablation={m:evaluate(baseline,val,ablate=m) for m in ("cfp","oct")} if args.modalities == "both" else {}
    atomic_json(out/"warmup_report.json",warm|{"input_occlusion":ablation})
    del baseline;torch.cuda.empty_cache()
    results={}
    modes=("baseline","radon","scrambled","self") if args.modalities == "both" else ("baseline",)
    for mode in modes:
        g=PilotGraph(mode,args.seed,device,backbone="resnet18",handoff_ratio=args.handoff_ratio,cfp_size=args.cfp_size,modalities=args.modalities)
        _,results[mode]=run(g,mode,args.arm_epochs,state)
        del g;torch.cuda.empty_cache()
    # Paired bootstrap uses identical validation participants across all arms.
    baseline_p=np.load(out/"baseline_predictions.npz")
    comparisons={}
    for mode in modes[1:]:
        arm=np.load(out/(mode+"_predictions.npz"))
        assert np.array_equal(arm["ids"],baseline_p["ids"])
        rng=np.random.default_rng(710);delta=[]
        for _ in range(1000):
            ix=rng.integers(0,len(arm["y"]),len(arm["y"]))
            y=arm["y"][ix]
            if len(np.unique(y))<2:continue
            delta.append(roc_auc_score(y,arm["p"][ix])-roc_auc_score(y,baseline_p["p"][ix]))
        comparisons[mode]={"delta_auroc":results[mode]["auroc"]-results["baseline"]["auroc"],
                           "paired_bootstrap_95pct":np.quantile(delta,[.025,.975]).tolist()}
    report={"warmup":warm,"arms":results,"comparisons_to_continued_baseline":comparisons,
            "elapsed_minutes":(time.time()-start)/60,"gpu_hours":(time.time()-start)/3600,
            "peak_process_mib":peak_process,"test_used":False,
            "baseline_screen_passed":warm["auroc"]>=.60 and warm["train"]["auroc"]>=.70,
            "limitations":["single seed","small balanced validation cohort","same validation used for checkpoint selection",
                           "coarse uniformly sampled OCT volume","record-derived participant glaucoma label",
                           "inflated ImageNet weights are not OCT-pretrained weights","no independent test or clinical claim"]}
    atomic_json(out/"summary.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--batch",type=int,default=4);p.add_argument("--warmup-epochs",type=int,default=10)
    p.add_argument("--arm-epochs",type=int,default=5);p.add_argument("--seed",type=int,default=3407)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--max-minutes",type=float,default=60)
    p.add_argument("--handoff-ratio",type=float,default=.03125)
    p.add_argument("--cfp-size",type=int,choices=(96,224),default=96)
    p.add_argument("--modalities",choices=("both","cfp","oct"),default="both")
    main(p.parse_args())
