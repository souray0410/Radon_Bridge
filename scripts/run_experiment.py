"""Sequential, time-bounded ws execution with one checkout lock and immutable outputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument("--protocol",required=True);p.add_argument("--data",required=True)
p.add_argument("--output",required=True);p.add_argument("--lock",required=True)
args=p.parse_args()
protocol=json.loads(Path(args.protocol).read_text())
env=os.environ.copy();env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
start=time.monotonic();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
with open(args.lock,"a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if any(root.iterdir()): raise RuntimeError("Batch output must be new")
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2))
    for run in protocol["runs"]:
        remaining=protocol["max_minutes"]*60-(time.monotonic()-start)
        if remaining<=0: raise RuntimeError("Batch wall-clock budget exhausted")
        cmd=[sys.executable,"-m","radonbridge.pilot","--data",args.data,
             "--output",str(root/run["id"]),"--max-minutes",str(min(8,remaining/60))]
        for key,value in (protocol["common"]|{k:v for k,v in run.items() if k!="id"}).items():
            cmd.extend(["--"+key.replace("_","-"),str(value)])
        print(json.dumps({"starting":run["id"],"remaining_minutes":remaining/60}),flush=True)
        with (root/(run["id"]+".log")).open("w") as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        summary=json.loads((root/run["id"]/"summary.json").read_text())
        print(json.dumps({"finished":run["id"],"minutes":summary["elapsed_minutes"],
                          "peak_process_mib":summary["peak_process_mib"],
                          "auroc":{k:v["auroc"] for k,v in summary["arms"].items()}}),flush=True)
    (root/"complete.json").write_text(json.dumps({"minutes":(time.monotonic()-start)/60}))
