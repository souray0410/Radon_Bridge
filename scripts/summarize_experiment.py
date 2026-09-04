"""Aggregate fixed-last and selected outcomes; no participant data leave ws."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True)
args=p.parse_args();root=Path(args.root)
protocol=json.loads((root/"protocol.json").read_text())
rows=[];runs={}
for spec in protocol["runs"]:
    run=root/spec["id"];summary=json.loads((run/"summary.json").read_text())
    config=json.loads((run/"config.json").read_text());runs[spec["id"]]=summary
    for arm,result in summary["arms"].items():
        rows.append({"run":spec["id"],"arm":arm,"seed":config["seed"],
                     "cfp_size":config["cfp_size"],"modalities":config["modalities"],
                     "source_commit":config["source_commit"],"selected_epoch":result["best_epoch"],
                     "selected_auroc":result["auroc"],"fixed_last":result["fixed_last"],
                     "selected_train_auroc":result["train"]["auroc"]})

def compare(run_a,arm_a,run_b,arm_b):
    with np.load(root/run_a/(arm_a+"_last_predictions.npz")) as a, np.load(root/run_b/(arm_b+"_last_predictions.npz")) as b:
        assert np.array_equal(a["ids"],b["ids"]) and np.array_equal(a["y"],b["y"])
        y=a["y"];rng=np.random.default_rng(710);deltas=[]
        for _ in range(2000):
            ix=rng.integers(0,len(y),len(y))
            if len(np.unique(y[ix]))<2:continue
            deltas.append(roc_auc_score(y[ix],a["p"][ix])-roc_auc_score(y[ix],b["p"][ix]))
        return {"a":run_a+"/"+arm_a,"b":run_b+"/"+arm_b,
                "delta_auroc":float(roc_auc_score(y,a["p"])-roc_auc_score(y,b["p"])),
                "paired_bootstrap_95pct":np.quantile(deltas,[.025,.975]).tolist()}

comparisons=[]
for spec in protocol["runs"]:
    if spec["modalities"]=="both":
        for control in ("baseline","scrambled","self"):
            comparisons.append(compare(spec["id"],"radon",spec["id"],control))
for seed in (3407,3408):
    comparisons.append(compare(f"both224_s{seed}","baseline",f"both96_s{seed}","baseline"))
comparisons.append(compare("cfp224_s3407","baseline","cfp96_s3407","baseline"))
report={"rows":rows,"fixed_last_comparisons":comparisons,
        "total_run_minutes":sum(s["elapsed_minutes"] for s in runs.values()),
        "peak_process_mib":max(s["peak_process_mib"] for s in runs.values()),
        "test_used":False,"uncertainty":"Unadjusted exploratory participant bootstrap; shared validation cohort, two seeds only. Warmup still uses validation checkpoint selection."}
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
