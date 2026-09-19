"""Finite cross-attention A -> A+Radon / ordinary matched augmentation package."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, os, subprocess
from pathlib import Path

from radon_bridge.runtime.state import atomic_write_json
from radon_bridge.studies.cohort_case import sha
from radon_bridge.studies.cohort_augmentation import migrate


ATTENTION_NAME = "attention_seed3416"
CASE_IDS = ("host_continue", "host_radon", "host_linear")
COMPARISONS = [["host_radon","host_continue"],["host_linear","host_continue"],["host_radon","host_linear"]]
FIXED = dict(seed=3416, stage=3, channel_rank=32, M=32, S=64, kernel=3, group_count=1,
             real_batch=16, minimum_epochs=8, maximum_epochs=60, patience=6, min_delta=.001)


def read(path): return json.loads(Path(path).read_text())
def write(path, value): atomic_write_json(value, Path(path))
def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _attention_case(core):
    q=read(core/"queue.json")
    if q.get("schema")!="radon_small_cohort_core_v1" or q.get("test_used") is not False:
        raise ValueError("Core queue not accepted")
    case=next((c for c in q["cases"] if c["id"]=="attention"),None)
    if case is None: raise ValueError("Accepted attention case missing")
    trial=core/"trials"/case["name"]; receipt=read(trial/"accepted.json"); cfg=read(case["config"])
    if (receipt.get("configuration")!=cfg or receipt.get("converged_by_policy") is not True
            or receipt.get("test_used") is not False or receipt.get("best_epoch")!=7 or receipt.get("epochs_ran")!=13):
        raise ValueError("Attention acceptance identity changed")
    for name,digest in receipt["files"].items():
        if sha(trial/name)!=digest: raise ValueError("Attention artifact changed: "+name)
    status=read(core/"status.json"); audit=read(core/"independent_final_audit.json")
    if (status.get("state")!="complete" or status.get("accepted")!=6 or status.get("failed")
            or audit.get("passed") is not True or audit.get("test_used") is not False):
        raise ValueError("Core package acceptance changed")
    return q,case,trial,receipt,cfg


def native_state_identity(trial,cfg):
    """Compare every native tensor represented in the accepted A against original parents."""
    import torch
    frozen=torch.load(trial/"best.pt",map_location="cpu",weights_only=False)["model"]
    rows={}; exact=True
    for branch,ref in cfg["parents"].items():
        parent=torch.load(ref["path"],map_location="cpu",weights_only=False)["model"]
        total=changed=0; max_abs=0.0
        for module,state in parent.items():
            if module not in frozen: continue
            for key,value in state.items():
                if key not in frozen[module]: continue
                current=frozen[module][key]; total+=1
                if not torch.equal(value,current):
                    changed+=1; exact=False
                    if torch.is_tensor(value) and torch.is_tensor(current) and value.shape==current.shape and value.dtype.is_floating_point:
                        max_abs=max(max_abs,float((value-current).abs().max()))
        rows[branch]=dict(compared=total,different=changed,max_abs=max_abs)
    return dict(exact=exact,branches=rows)


def prepare(core_root, output, sequence_id, source_commit, framework_commit, source_root):
    core=Path(core_root); out=Path(output); source_root=Path(source_root)
    if out.exists(): raise FileExistsError(out)
    q,case,trial,receipt,cfg=_attention_case(core)
    out.mkdir(parents=True); (out/"configs").mkdir(); (out/"trials").mkdir()
    migrate(trial,out/"host")
    identity=native_state_identity(trial,cfg)
    if identity["exact"]:
        raise ValueError("Attention A unexpectedly equals native parents; package requires explicit basis decision")
    basis_cfg=dict(host_checkpoint=dict(path=str((trial/"best.pt").resolve()),sha256=receipt["files"]["best.pt"]))
    write(out/"host_basis_config.json",basis_cfg)
    protocol=dict(schema="radon_attention_augmentation_protocol_v1",study_kind="existing_method_augmentation",
        sequence_id=sequence_id,source_commit=source_commit,framework_commit=framework_commit,
        source_core=str(core),source_core_status_sha256=sha(core/"status.json"),
        source_core_audit_sha256=sha(core/"independent_final_audit.json"),
        source_attention_receipt_sha256=sha(trial/"accepted.json"),
        source_attention_prediction_sha256=receipt["files"]["selected_predictions.npz"],
        source_attention_best_sha256=receipt["files"]["best.pt"],source_attention_best_epoch=7,
        source_attention_stop_epoch=13,source_attention_config=cfg,native_parent_state_identity=identity,
        data=q["data"],test_used=False,fixed=FIXED,comparisons=COMPARISONS,
        basis_policy="fresh uncentered SVD from frozen attention best7 train Stage3 pre-write features",
        host_semantics="project cross-attention Stage3 adapter, d256 heads4; not external author-system reproduction",
        question="Does matched Radon addition improve an already trained cross-attention communication host beyond continued training and ordinary communication?")
    write(out/"protocol.json",protocol)
    old_dep=read(core/"dependencies_acceptance.json")
    if old_dep.get("passed") is not True or old_dep.get("test_used") is not False:
        raise ValueError("Core dependency receipt changed")
    refs={str(trial/"accepted.json"):sha(trial/"accepted.json"),str(trial/"best.pt"):receipt["files"]["best.pt"],
          str(trial/"selected_predictions.npz"):receipt["files"]["selected_predictions.npz"],
          str(out/"host/manifest.json"):sha(out/"host/manifest.json"),str(out/"host/model.pt"):sha(out/"host/model.pt")}
    prep=dict(schema="radon_attention_augmentation_preparation_v1",passed=True,test_used=False,
        protocol_sha256=sha(out/"protocol.json"),host_basis_config_sha256=sha(out/"host_basis_config.json"),
        references=refs,data_files=old_dep["data_files"])
    write(out/"preparation_acceptance.json",prep)
    return protocol


def _gpu_lock():
    import psutil
    if psutil.virtual_memory().available < .15*psutil.virtual_memory().total: raise MemoryError("Host reserve below 15 percent")
    device=os.environ.get("CUDA_VISIBLE_DEVICES")
    if not device or "," in device: raise ValueError("One explicit CUDA device required")
    used,total=map(int,subprocess.check_output(["nvidia-smi","-i",device,"--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],text=True).strip().split(","))
    if total-used < 20*1024: raise MemoryError("Need 10GiB worker plus 10GiB reserve")
    uuid=subprocess.check_output(["nvidia-smi","-i",device,"--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    root=Path(os.environ["RESEARCH_GPU_LOCK_ROOT"]);root.mkdir(parents=True,exist_ok=True)
    handle=(root/(uuid+".lock")).open("a");fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);return handle


def fit_basis(root):
    root=Path(root); protocol=read(root/"protocol.json")
    if (root/"basis_fit/summary.json").exists(): return validate_basis(root)
    from radon_bridge.methods.host_basis import main
    lock=_gpu_lock()
    try: main(read(root/"host_basis_config.json"),root/"basis_fit",protocol["data"])
    finally: lock.close()
    return validate_basis(root)


def validate_basis(root):
    root=Path(root); protocol=read(root/"protocol.json"); summary=read(root/"basis_fit/summary.json")
    cfg=read(root/"host_basis_config.json")
    if (summary.get("state")!="complete" or summary.get("passed") is not True or summary.get("test_used") is not False
            or summary.get("provenance",{}).get("host_checkpoint")!=cfg["host_checkpoint"]
            or summary.get("provenance",{}).get("training_participants")!=1264
            or summary.get("provenance",{}).get("feature_position")!="intact host stage3 before communication write-back"):
        raise ValueError("Attention host basis summary changed")
    if set(summary.get("basis_files",{}))!={"cfp_stage3","oct_stage3"}: raise ValueError("Attention basis nodes changed")
    from radon_bridge.methods.basis import _load_basis, BASIS_VERSION
    for key,ref in summary["basis_files"].items():
        q,values,meta=_load_basis(ref["path"],ref["sha256"])
        if meta["version"]!=BASIS_VERSION or meta["source_key"]!=key or meta["seed"]!=3416 or meta["centered"] is not False:
            raise ValueError("Attention basis metadata changed")
        if q.shape!=(256,256): raise ValueError("Attention basis shape changed")
    return summary


def finalize(root, source_root):
    root=Path(root); source_root=Path(source_root); protocol=read(root/"protocol.json"); basis=validate_basis(root)
    if (root/"queue.json").exists(): return read(root/"queue.json")
    host=read(root/"host/manifest.json"); base=copy.deepcopy(host["configuration"]); bases=basis["basis_files"]
    href=dict(path=str((root/"host/manifest.json").resolve()),sha256=sha(root/"host/manifest.json"))
    configs={}; cases=[]
    for key,mode in [("host_continue",None),("host_radon","radon"),("host_linear","linear_resample")]:
        cfg=copy.deepcopy(base);cfg["name"]=key+"_seed3416";cfg["augmentation_host"]=href
        if mode:
            cfg["bridges"].append(dict(nodes=["cfp_stage3","oct_stage3"],M=32,S=64,rho=.125,mode=mode,
                compression="fixed_svd_channel",basis_files=copy.deepcopy(bases),kernel_size=3,parallel_to=0))
        path=root/"configs"/(cfg["name"]+".json");write(path,cfg);configs[str(path)]=sha(path)
        cases.append(dict(id=key,name=cfg["name"],config=str(path),resource=key,seed=3416,
            provenance="same accepted cross-attention host; new matched continuation"))
    host_summary=dict(method="project cross-attention Stage3 adapter",display_name="交叉注意力",best_epoch=7,
        source_model_sha256=protocol["source_attention_best_sha256"],native_parent_state_exact=False,
        limitation_tag="project_cross_attention_adapter_not_author_system",
        system_boundary="本项目cross-attention适配，不声称复现外部作者完整系统",
        basis_policy="冻结attention best7原生状态与原SVD父状态不一致；从冻结A的train Stage3 pre-write features重新拟合共享未中心化SVD基")
    queue=dict(schema="radon_small_cohort_core_v1",study_kind="existing_method_augmentation",
        sequence_id=protocol["sequence_id"],data=protocol["data"],test_used=False,cases=cases,
        comparisons=COMPARISONS,host_summary=host_summary)
    write(root/"queue.json",queue)
    prep=read(root/"preparation_acceptance.json");refs=dict(prep["references"])
    refs[str(root/"basis_fit/summary.json")]=sha(root/"basis_fit/summary.json")
    for ref in bases.values(): refs[ref["path"]]=ref["sha256"]
    deps=dict(passed=True,test_used=False,queue_sha256=sha(root/"queue.json"),config_files=configs,
        references=refs,data_files=prep["data_files"],preparation_acceptance_sha256=sha(root/"preparation_acceptance.json"))
    write(root/"dependencies_acceptance.json",deps)
    framework=read(source_root/"framework.lock.json")
    files={}
    for p in sorted((source_root/"src/radon_bridge").rglob("*.py")): files[str(p.resolve())]=sha(p)
    for item in framework["files"]:
        p=source_root/item["local_path"]
        if sha(p)!=item["sha256"]: raise ValueError("Framework source lock changed")
        files[str(p.resolve())]=item["sha256"]
    write(root/"code_acceptance.json",dict(passed=True,passed_cpu=True,test_used=False,
        source_commit=protocol["source_commit"],framework_commit=protocol["framework_commit"],files=files))
    return queue


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare");a.add_argument("--core",required=True);a.add_argument("--output",required=True);a.add_argument("--sequence-id",required=True);a.add_argument("--source-commit",required=True);a.add_argument("--framework-commit",required=True);a.add_argument("--source-root",required=True)
    a=sub.add_parser("fit-basis");a.add_argument("--root",required=True)
    a=sub.add_parser("finalize");a.add_argument("--root",required=True);a.add_argument("--source-root",required=True)
    x=p.parse_args()
    if x.cmd=="prepare": result=prepare(x.core,x.output,x.sequence_id,x.source_commit,x.framework_commit,x.source_root)
    elif x.cmd=="fit-basis": result=fit_basis(x.root)
    else: result=finalize(x.root,x.source_root)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__": main()
