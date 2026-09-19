import json
from pathlib import Path
import numpy as np
from radon_bridge.analysis.cohort_report import report
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.studies.cohort_case import sha


def test_attention_augmentation_report_is_host_generic(tmp_path):
    y=np.arange(296)%2;prob=np.stack([1-y,y],axis=1)*.8+.1
    cases=[]
    for key in ("host_continue","host_radon","host_linear"):
        path=tmp_path/"trials"/key;path.mkdir(parents=True)
        cfg={"seed":3416};spec=tmp_path/(key+".json");spec.write_text(json.dumps(cfg))
        np.savez(path/"selected_predictions.npz",ids=np.array([str(i) for i in range(296)]),y=y,cfp=prob,oct=prob)
        (path/"best.pt").write_bytes((key+" model").encode())
        receipt=dict(configuration=cfg,test_used=False,converged_by_policy=True,
            files={n:sha(path/n) for n in ("selected_predictions.npz","best.pt")},
            best_epoch=1,epochs_ran=8,selected_validation=dict(tasks={k:classification_metrics(y,prob) for k in ("cfp","oct")}))
        (path/"accepted.json").write_text(json.dumps(receipt))
        cases.append(dict(id=key,name=key,config=str(spec),resource=key,seed=3416,provenance="fixture"))
    q=dict(schema="radon_small_cohort_core_v1",sequence_id="attention_fixture",study_kind="existing_method_augmentation",
        cases=cases,comparisons=[["host_radon","host_continue"],["host_linear","host_continue"],["host_radon","host_linear"]],
        host_summary=dict(display_name="交叉注意力",best_epoch=7,limitation_tag="project_cross_attention_adapter_not_author_system",
            system_boundary="本项目cross-attention适配，不声称复现外部作者完整系统",
            basis_policy="冻结attention best7后fresh SVD"))
    (tmp_path/"queue.json").write_text(json.dumps(q))
    value=report(tmp_path);text=(tmp_path/"publication/README.md").read_text()
    assert value["study_kind"]=="existing_method_augmentation"
    assert "交叉注意力继续训练" in text and "交叉注意力＋Radon桥" in text
    assert "第一阶段选中第7轮" in text and "fresh SVD" in text
    assert "第一阶段选择第0轮" not in text
    assert "project_cross_attention_adapter_not_author_system" in value["limitations"]
