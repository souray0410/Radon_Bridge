"""Bounded UKB pilot extraction: train/validation only, no source writes."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

CACHE_SCHEMA = "fullspan32_cfp96_oct96_v1"


def prepare(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.labels, dtype={"participant_id": str})
    frame = frame[frame.split.isin(["train", "validation"])].copy()
    frame["order"] = frame.participant_id.map(lambda x: hashlib.sha256(("rb-pilot-3407:" + x).encode()).hexdigest())
    chosen = []
    for split, count in [("train", args.train), ("validation", args.validation)]:
        part = frame[frame.split == split].sort_values("order")
        chosen.append(pd.concat([part[part.label_id == y].head(count // 2) for y in (0, 1)]))
    selected = pd.concat(chosen).sort_values(["split", "order"])
    if selected.participant_id.duplicated().any(): raise ValueError("Duplicate participant")
    root.joinpath("selected.csv").write_text(selected.to_csv(index=False))
    records = selected.to_dict("records")
    start = time.time()
    def one(row):
        key = row["order"]
        dst = root / (key + ".npz")
        if dst.exists():
            with np.load(dst, allow_pickle=False) as a:
                if str(a["schema"]) != CACHE_SCHEMA or a["cfp"].shape != (2,3,96,96) or a["oct"].shape != (2,1,32,96,96):
                    raise ValueError("Cache mismatch")
            return
        cfps, octs, indices, counts, native_sizes = [], [], [], [], []
        for eye in ("left", "right"):
            cfp_path = Path(args.image_root) / row[eye + "_fundus_path"]
            # Existing exporter uses <archive-parent>/<archive-stem>/<slice>.
            archive = Path(args.source_root) / Path(row[eye + "_oct_path"]).parent.with_suffix(".zip")
            with Image.open(cfp_path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im); ys, xs = np.where(arr.max(-1) > 8)
                if not len(xs): raise ValueError("Empty fundus")
                im = im.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
                side = max(im.size); padded = Image.new("RGB", (side,side))
                padded.paste(im, ((side-im.width)//2, (side-im.height)//2))
                cfps.append(np.asarray(padded.resize((96,96), Image.Resampling.BILINEAR)).transpose(2,0,1))
            with zipfile.ZipFile(archive) as z:
                members = [i for i in z.infolist() if Path(i.filename).suffix.lower() == ".png"]
                numbers = [int(re.search(r"_(\d+)\.[^.]+$", i.filename).group(1)) for i in members]
                ordered = sorted(zip(numbers, members))
                ns = [i[0] for i in ordered]
                if len(ns) != len(set(ns)) or len(ns) < 32 or np.any(np.diff(ns) != 1):
                    raise ValueError("Incomplete or duplicate OCT slices")
                pick = np.round(np.linspace(0,len(ordered)-1,32)).astype(int)
                planes = []
                native_size = None
                for k in pick:
                    with z.open(ordered[k][1]) as stream:
                        with Image.open(stream) as im:
                            if native_size is None: native_size=im.size
                            if min(im.size)<32 or im.size!=native_size:raise ValueError("Invalid or inconsistent within-volume OCT dimensions")
                            planes.append(np.asarray(im.convert("L").resize((96,96),Image.Resampling.BILINEAR)))
                octs.append(np.stack(planes)[None]); indices.append(np.asarray(ns)[pick]); counts.append(len(ns));native_sizes.append(native_size)
        tmp = dst.with_suffix(".partial")
        with tmp.open("wb") as f:
            np.savez(f, cfp=np.stack(cfps), oct=np.stack(octs), schema=CACHE_SCHEMA,
                     slice_numbers=np.stack(indices), native_slice_counts=counts,native_sizes=native_sizes)
        tmp.replace(dst)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(one,records),1):
            if i % 16 == 0: print(json.dumps({"prepared":i,"total":len(records),"seconds":round(time.time()-start,1)}),flush=True)
    report = {"schema":CACHE_SCHEMA,"participants":len(selected),
              "split_counts":{str(k):int(v) for k,v in selected.groupby(["split","label_id"]).size().items()},
              "patient_overlap":0,"test_used":False,"seconds":time.time()-start,
              "sampling":"32 ordered B-scans uniformly over the full 128-slice range; no spacing claim",
              "labels_sha256":hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
              "selected_sha256":hashlib.sha256(root.joinpath("selected.csv").read_bytes()).hexdigest()}
    shapes={}
    for row in records:
        with np.load(root/(row["order"]+".npz"),allow_pickle=False) as a:
            # Earlier caches passed an explicit 512x650 check for every decoded plane.
            for shape in a["native_sizes"] if "native_sizes" in a else [(512,650),(512,650)]:
                key="x".join(map(str,shape));shapes[key]=shapes.get(key,0)+1
    report["native_width_height_counts"]=shapes
    root.joinpath("audit.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


class PairedDataset(Dataset):
    def __init__(self, root, split):
        self.root=Path(root)
        f=pd.read_csv(self.root/"selected.csv",dtype={"participant_id":str})
        self.rows=f[f.split==split].to_dict("records")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with np.load(self.root/(row["order"]+".npz"),allow_pickle=False) as z:
            c=torch.from_numpy(z["cfp"].copy()).float()/255
            o=torch.from_numpy(z["oct"].copy()).float()/255
        c=(c-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)
        o=(o-.5)/.25
        return c,o,int(row["label_id"]),row["order"]


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("labels","image-root","source-root","output"):p.add_argument("--"+name,required=True)
    p.add_argument("--train",type=int,default=256);p.add_argument("--validation",type=int,default=128)
    p.add_argument("--workers",type=int,default=3)
    prepare(p.parse_args())
