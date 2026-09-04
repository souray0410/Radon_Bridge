"""Render aggregate results only; participant-level inputs are never needed."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True)
args=p.parse_args();report=json.loads(Path(args.input).read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
colors={"baseline":"#273c75","radon":"#e17055","scrambled":"#718093","self":"#44a699"}
for seed,marker in ((3407,"o"),(3408,"s")):
    for arm,color in colors.items():
        values=[next(r["fixed_last"]["auroc"] for r in report["rows"]
                     if r["run"]==f"both{size}_s{seed}" and r["arm"]==arm) for size in (96,224)]
        axes[0].plot([96,224],values,marker=marker,color=color,
                     linestyle="-" if seed==3407 else "--",label=arm if seed==3407 else None)
axes[0].set(title="Joint models: two seeds",xlabel="CFP input side (pixels)",ylabel="Validation AUROC")
axes[0].set_xticks([96,224]);axes[0].legend(frameon=False,fontsize=8)
names=["cfp96_s3407","cfp224_s3407","oct_s3407","both96_s3407","both224_s3407"]
vals=[next(r["fixed_last"]["auroc"] for r in report["rows"] if r["run"]==name and r["arm"]=="baseline") for name in names]
axes[1].bar(range(5),vals,color=["#718093","#273c75","#44a699","#a29bfe","#6c5ce7"],width=.65)
axes[1].set_xticks(range(5),["CFP 96","CFP 224","OCT","Joint 96","Joint 224"],rotation=25)
axes[1].set(title="Independently trained baselines: seed 3407",ylabel="Validation AUROC")
for ax in axes:
    ax.axhline(.5,color="black",linewidth=.7,linestyle=":")
    ax.set_ylim(.45,.85);ax.spines[["top","right"]].set_visible(False)
fig.suptitle("UKB pilot: fixed final epoch, 256 train / 128 validation participants",fontsize=11)
fig.savefig(args.output,dpi=180)
