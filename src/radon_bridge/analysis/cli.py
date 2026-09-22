"""Read-only mechanism diagnostics and fixed-recipe representation probes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from radon_bridge.models.model import PilotGraph as _PilotGraph
from functools import partial
PilotGraph = partial(_PilotGraph, head_mode="shared_legacy")  # historical outputs only
from radon_bridge.data.dataset import PairedDataset


def main(args):
    torch.set_num_threads(3)
    from mhd_models.scheduling.gpu_budget import configure_allocator
    configure_allocator(0)
    def loader(split):return DataLoader(PairedDataset(args.data,split,config.get("cfp_size",96)),batch_size=4,num_workers=2)
    report={}
    with torch.no_grad():
        for run_name in args.runs:
            run=Path(run_name);config=json.loads((run/"config.json").read_text())
            if config.get("modalities","both") != "both": raise ValueError("Diagnostics require a joint run")
            ratio=config.get("handoff_ratio",.25)
            for ck in ("best","last"):
                g=PilotGraph("radon",device="cuda",backbone="resnet18",handoff_ratio=ratio,cfp_size=config.get("cfp_size",96))
                state=torch.load(run/f"radon_{ck}.pt",map_location="cpu",weights_only=False)
                g.load_state(state["model"]);g.graph.eval()
                mixer=g.modules_by_name()["projection_mixer"];n=mixer.h1
                deltas={"cfp":[],"oct":[]};diff=[]
                for c,o,y,_ in loader("validation"):
                    c,o,y=c.cuda(),o.cuda(),y.cuda()
                    logits,_=g.forward(c,o,y);p=logits.softmax(1)[:,1]
                    for name in deltas:
                        f=g.by_name[name+"_stage3"].feature_message.current_state
                        delta=g.by_name[name+"_delta"].feature_message.current_state
                        deltas[name].extend((delta.flatten(1).norm(dim=1)/f.flatten(1).norm(dim=1).clamp_min(1e-8)).cpu().tolist())
                    mixer.mask[:n,n:]=0;mixer.mask[n:,:n]=0
                    z,_=g.forward(c,o,y);diff.extend((p-z.softmax(1)[:,1]).abs().cpu().tolist())
                    mixer.mask.fill_(1)
                report[run.name+"_"+ck]={
                    "relative_delta_median":{k:float(np.median(v)) for k,v in deltas.items()},
                    "cross_disable_mean_probability_change":float(np.mean(diff)),
                    "cross_disable_max_probability_change":float(np.max(diff)),
                    "mixer_norm":float(mixer.conv.weight.norm()),
                    "cross_mixer_norm":float(torch.cat((mixer.conv.weight[:n,n:].flatten(),mixer.conv.weight[n:,:n].flatten())).norm()),
                    "epoch":state["epoch"]}
                del g,state;torch.cuda.empty_cache()
        # These are probes of jointly trained features, NOT independently
        # trained unimodal clinical baselines. C=0.01 is fixed before fitting.
        run=Path(args.runs[-1]);g=PilotGraph(device="cuda",backbone="resnet18",cfp_size=config.get("cfp_size",96))
        g.load_state(torch.load(run/"warmup_best.pt",map_location="cpu",weights_only=False)["model"]);g.graph.eval()
        features={}
        for split in ("train","validation"):
            cc=[];oo=[];yy=[]
            for c,o,y,_ in loader(split):
                g.forward(c.cuda(),o.cuda(),y.cuda())
                cc.append(g.by_name["cfp_participant"].feature_message.current_state.cpu().numpy())
                oo.append(g.by_name["oct_participant"].feature_message.current_state.cpu().numpy());yy.extend(y.numpy())
            features[split]={"cfp":np.concatenate(cc),"oct":np.concatenate(oo),"y":np.asarray(yy)}
            features[split]["concat"]=np.concatenate((features[split]["cfp"],features[split]["oct"]),axis=1)
        del g;torch.cuda.empty_cache()
    probes={}
    for mode in ("cfp","oct","concat"):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000,solver="lbfgs"))
        model.fit(features["train"][mode],features["train"]["y"])
        probes[mode]={}
        for split in ("train","validation"):
            p=model.predict_proba(features[split][mode])[:,1];y=features[split]["y"]
            probes[mode][split]={"auroc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p))}
    report["fixed_logistic_probes"]={"C":.01,"source":str(run),"results":probes,
                                      "interpretation":"probes of jointly trained features; not independently trained unimodal models"}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--runs",nargs="+",required=True);p.add_argument("--output",required=True)
    main(p.parse_args())
