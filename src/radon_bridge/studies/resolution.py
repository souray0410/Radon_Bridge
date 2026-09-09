"""Build higher-resolution CFP from originals, reusing the unchanged OCT cache."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from radon_bridge.data.dataset import read_cfp


def main(args):
    root=Path(args.data); dst=root/"cfp224"; dst.mkdir(exist_ok=True)
    manifest=root/"selected.csv"
    rows=pd.read_csv(manifest,dtype={"participant_id":str}).to_dict("records")
    def one(row):
        images=[]; source_hashes=[]
        with np.load(root/(row["order"]+".npz"),allow_pickle=False) as old:
            for i,eye in enumerate(("left","right")):
                path=Path(args.image_root)/row[eye+"_fundus_path"]
                # Verify the original crop/normalization precursor is unchanged.
                if not np.array_equal(read_cfp(path,96),old["cfp"][i]):
                    raise ValueError("Original-to-96 preprocessing changed")
                images.append(read_cfp(path,224))
                source_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        array=np.stack(images); target=dst/(row["order"]+".npy")
        if target.exists():
            if not np.array_equal(np.load(target,allow_pickle=False),array):
                raise ValueError("Existing high-resolution cache differs")
        else:
            temp=target.with_suffix(".partial")
            with temp.open("wb") as f: np.save(f,array,allow_pickle=False)
            temp.replace(target)
        return hashlib.sha256(array.tobytes()).hexdigest(),source_hashes
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes=list(pool.map(one,rows))
    audit={"participants":len(rows),"cfp_shape":[2,3,224,224],
           "source":"original exported CFP, same ROI crop and padding; bilinear resize",
           "oct":"unchanged original pilot NPZ cache",
           "all_96_reconstructions_pixel_identical":True,
           "selected_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
           "ordered_source_and_cache_digest":hashlib.sha256(json.dumps(hashes).encode()).hexdigest()}
    (dst/"audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True)
    p.add_argument("--image-root",required=True);p.add_argument("--workers",type=int,default=3)
    main(p.parse_args())
