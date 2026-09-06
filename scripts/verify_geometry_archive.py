"""Independent restore gate before deleting any old work-disk timestamp."""
import json,time
from pathlib import Path
import numpy as np
import torch
from radonbridge.artifacts import ARCHIVE,sha256
from radonbridge.data import PairedDataset
from radonbridge.model import PilotGraph
from radonbridge.experiment import evaluate
from scripts.geometry_evidence import read,write,force_archive,dependencies,archive_path

def main():
    manifest=ARCHIVE/'archive_manifest.json';assert read(manifest)['state']=='verified'
    parents,bases=dependencies()
    old=read(ARCHIVE/'history/runs/2026_09_05_22_42_53/manifest.json')['rows']
    references=[next(r for r in old if r['seed']==s and r['arm']=='svd_radon' and r['rho']==.125) for s in (3416,3417,3418)]
    fusion=read(ARCHIVE/'history/runs/2026_09_06_08_37_46/manifest.json')['rows']
    references.extend(next(r for r in fusion if r['seed']==3416 and r['backbone_lr']==3e-5 and r['arm']==a) for a in ('concat_mlp','mmtm_r4','attention_d128'))
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    data=PairedDataset(str(ARCHIVE/'history/cache/full1264_296'),'validation',224);assert len(data)==296
    reports=[];out=ARCHIVE/'restore_checks';out.mkdir(exist_ok=True)
    for i,r in enumerate(references):
        path=archive_path(r['directory']);s=read(path/'summary.json');cfg=force_archive(s['configuration'])
        assert all(sha256(path/n)==v for n,v in r['accepted_hashes'].items())
        g=PilotGraph(seed=cfg['seed'],bridge_configs=cfg['bridges'],task_fusion=cfg.get('task_fusion'),device='cuda')
        saved=torch.load(path/'selected.pt',map_location='cpu',weights_only=False);g.load_complete_state(saved['model']);del saved
        g.graph.eval();pred=out/f'restored_{i}.npz';metrics=evaluate(g,data,16,cfg['seed'],pred)
        with np.load(pred,allow_pickle=False) as a,np.load(path/'selected_predictions.npz',allow_pickle=False) as b:
            assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['y'],b['y'])
            assert all(np.allclose(a[k],b[k],atol=1e-6,rtol=1e-5) for k in g.output_names)
        assert all(metrics['tasks'][k]['macro_f1']==s['selected']['tasks'][k]['macro_f1'] for k in g.output_names)
        reports.append(dict(directory=str(path),selected_sha256=sha256(path/'selected.pt'),strict_load=True,predictions_equal=True,statistics_equal=True))
        write(out/'progress.json',dict(completed=len(reports),total=len(references)))
        del g;torch.cuda.empty_cache()
    unresolved=[e for e in read(manifest)['entries'] if e.get('external_dependency') and not Path(e['external_dependency']).exists()]
    assert not unresolved,unresolved
    write(ARCHIVE/'restore_acceptance.json',dict(passed=True,archive_manifest_sha256=sha256(manifest),
          all_runtime_dependencies_resolved=True,strict_load_and_predictions=True,references=reports,
          parent_checkpoints=parents,basis_files=bases,test_used=False,raw_UKB_included=False))

if __name__=='__main__':main()
