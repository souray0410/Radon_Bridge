import json
import numpy as np
import pytest
from radon_bridge.analysis.cohort_report import report
from radon_bridge.evaluation.metrics import classification_metrics
from radon_bridge.studies.cohort_case import sha


def test_partial_publication_replay_and_tamper(tmp_path):
    y=np.arange(296)%2;p=np.stack([1-y,y],axis=1)*.8+.1
    case=tmp_path/'trials'/'none_seed3416';case.mkdir(parents=True)
    cfg={'seed':3416};spec=tmp_path/'spec.json';spec.write_text(json.dumps(cfg))
    np.savez(case/'selected_predictions.npz',ids=np.array([str(i) for i in range(296)]),y=y,cfp=p,oct=p)
    (case/'best.pt').write_bytes(b'test_model')
    files={name:sha(case/name) for name in ('selected_predictions.npz','best.pt')}
    receipt=dict(configuration=cfg,test_used=False,converged_by_policy=True,files=files,best_epoch=1,epochs_ran=8,
        selected_validation=dict(tasks={k:classification_metrics(y,p) for k in ('cfp','oct')}))
    (case/'accepted.json').write_text(json.dumps(receipt))
    q=dict(sequence_id='test',cases=[dict(id='none',name='none_seed3416',config=str(spec),provenance='accepted reference'),dict(id='svd',name='missing',config='absent',provenance='new')])
    (tmp_path/'queue.json').write_text(json.dumps(q))
    r=report(tmp_path);assert not r['complete'];assert r['results'][0]['mean_f1']==1
    dest=tmp_path/'publication/current.json';before=dest.read_bytes();stamp=dest.stat().st_mtime_ns
    report(tmp_path);assert dest.read_bytes()==before and dest.stat().st_mtime_ns==stamp
    (case/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Evidence changed'):report(tmp_path)
    assert dest.read_bytes()==before


def test_channel_package_uses_declared_contrasts_and_cfp_label(tmp_path):
    y=np.arange(296)%2;prob=np.stack([1-y,y],axis=1)*.8+.1
    cases=[]
    for key in ('svd','linear','qr','qr_linear','learned','learned_linear'):
        path=tmp_path/'trials'/key;path.mkdir(parents=True)
        cfg={'seed':3416};spec=tmp_path/(key+'.json');spec.write_text(json.dumps(cfg))
        np.savez(path/'selected_predictions.npz',ids=np.array([str(i) for i in range(296)]),y=y,cfp=prob,oct=prob)
        (path/'best.pt').write_bytes(b'model')
        receipt=dict(configuration=cfg,test_used=False,converged_by_policy=True,
            files={n:sha(path/n) for n in ('selected_predictions.npz','best.pt')},best_epoch=1,epochs_ran=8,
            selected_validation=dict(tasks={k:classification_metrics(y,prob) for k in ('cfp','oct')}))
        (path/'accepted.json').write_text(json.dumps(receipt))
        cases.append(dict(id=key,name=key,config=str(spec),provenance='fixture'))
    contrasts=[['svd','qr'],['qr','qr_linear'],['learned','learned_linear']]
    (tmp_path/'queue.json').write_text(json.dumps(dict(sequence_id='channel_fixture',study_kind='channel_compression',cases=cases,comparisons=contrasts)))
    r=report(tmp_path)
    assert r['complete'] and len(r['comparisons']['contrasts'])==3
    assert [(c['method'],c['reference']) for c in r['comparisons']['contrasts']]==[tuple(v) for v in contrasts]
    text=(tmp_path/'publication/README.md').read_text()
    assert 'CFP分支F1' in text and '随机QR' in text and '3项同时' in text


def test_grouped_completed_render_has_current_language_and_readable_sources(tmp_path):
    y=np.arange(296)%2;prob=np.stack([1-y,y],axis=1)*.8+.1
    references=[];cases=[]
    for group in (1,2,4,8,16):
        for mode in ('radon','linear_resample'):
            key=f'grouped_g{group}_{mode}';path=tmp_path/'trials'/key;path.mkdir(parents=True)
            cfg={'seed':3416};spec=tmp_path/(key+'.json');spec.write_text(json.dumps(cfg))
            np.savez(path/'selected_predictions.npz',ids=np.array([str(i) for i in range(296)]),y=y,cfp=prob,oct=prob)
            (path/'best.pt').write_bytes((key+' model').encode())
            receipt=dict(configuration=cfg,test_used=False,converged_by_policy=True,
                files={n:sha(path/n) for n in ('selected_predictions.npz','best.pt')},best_epoch=0 if mode=='linear_resample' else 1,epochs_ran=8,
                selected_validation=dict(tasks={k:classification_metrics(y,prob) for k in ('cfp','oct')}))
            (path/'accepted.json').write_text(json.dumps(receipt))
            item=dict(id=key,name=key,config=str(spec),provenance='machine provenance')
            if group==1:
                item['trial']=str(path);references.append(item)
            else:cases.append(item)
    q=dict(sequence_id='grouped_fixture',study_kind='grouped_linear',groups=[1,2,4,8,16],references=references,cases=cases,
        comparisons=[[f'grouped_g{g}_radon',f'grouped_g{g}_linear_resample'] for g in (1,2,4,8,16)])
    (tmp_path/'queue.json').write_text(json.dumps(q))
    status=dict(state='complete',planned=8,accepted=8,active={},failed={},updated_at=1)
    (tmp_path/'status.json').write_text(json.dumps(status))
    audit=dict(passed=True,test_used=False,queue_sha256=sha(tmp_path/'queue.json'),status_sha256=sha(tmp_path/'status.json'))
    (tmp_path/'independent_final_audit.json').write_text(json.dumps(audit))
    r=report(tmp_path);assert r['complete'] and len(r['results'])==10
    text=(tmp_path/'publication/README.md').read_text()
    assert '覆盖与后续机制' in text and '未完成的分组及机制' not in text
    assert '完整后自动生成' not in text and '本页已基于296名开发集参与者生成10,000次配对bootstrap' in text
    assert 'G=1严格复用已接受核心结果' in text and '新增单种子3416匹配执行' in text
    assert 'G表示桥内分组数' in text and '只有同一G下Radon与普通线性重采样是参数匹配的几何对照' in text
    assert 'machine provenance' not in text
