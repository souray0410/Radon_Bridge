"""Export only aggregate metrics; participant predictions stay on ws02."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from radonbridge.metrics import classification_metrics
from radonbridge.experiment import write_json


def main(root):
    root=Path(root); records=[]
    ledger=json.loads((root/'ledger.json').read_text())
    accounting={j['id']:j for j in ledger['jobs']}
    for path in sorted(root.glob('*/summary.json')):
        result=json.loads(path.read_text())
        if 'fixed_last' not in result:continue
        cfg=result['configuration'];identifier=path.parent.name
        with np.load(path.parent/'last_predictions.npz',allow_pickle=False) as a:
            for task in ['cfp','oct']:
                recalculated=classification_metrics(a['y'],a[task])
                expected=result['fixed_last']['tasks'][task]
                for key in ['macro_f1','macro_precision','macro_recall']:
                    assert abs(recalculated[key]-expected[key])<1e-12
                records.append({'trial':identifier,'seed':cfg['seed'],'task':task,'epochs':cfg['epochs'],
                    'macro_f1':expected['macro_f1'],'macro_precision':expected['macro_precision'],
                    'macro_recall':expected['macro_recall'],'auroc':expected.get('auroc'),
                    'parameters':result['parameters'],'gpu_seconds':accounting.get(identifier,{}).get('gpu_seconds',result['seconds']),
                    'peak_process_mib':accounting.get(identifier,{}).get('sampled_peak_process_mib'),
                    'confusion_matrix':expected['confusion_matrix']})
    baseline={(r['seed'],r['task']):r['macro_f1'] for r in records if r['trial'].endswith('_independent')}
    for row in records:
        base=baseline.get((row['seed'],row['task']));row['delta_macro_f1']=row['macro_f1']-base if base is not None else None
    aggregate=[]
    for arm in ['independent','radon','self','pooled']:
        for task in ['cfp','oct']:
            rows=[r for r in records if r['trial'].startswith('confirm_') and r['trial'].endswith('_'+arm) and r['task']==task]
            if rows:
                f1=np.array([r['macro_f1'] for r in rows]);delta=[r['delta_macro_f1'] for r in rows if r['delta_macro_f1'] is not None]
                aggregate.append({'arm':arm,'task':task,'seeds_completed':len(rows),'mean_macro_f1':float(f1.mean()),
                    'std_macro_f1':float(f1.std(ddof=1)) if len(rows)>1 else None,'mean_delta_macro_f1':float(np.mean(delta)) if delta else None})
    write_json(root/'aggregate.json',{'trials':records,'confirmation':aggregate,'predictions_recomputed':True,
                                    'test_used':False,'interpretation':'exploratory development-set evidence'})
    if records:
        with (root/'aggregate.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Rhythm Bridge exploratory results','','Fixed last epoch; macro-F1 per branch. Development set was used in prior task screening; test remains untouched.','',
           '| Trial | Branch | Macro-F1 (%) | Change (pp) | Peak process (MiB) |','|---|---|---:|---:|---:|']
    for r in records:
        delta='—' if r['delta_macro_f1'] is None else f"{100*r['delta_macro_f1']:+.2f}"
        lines.append(f"| {r['trial']} | {r['task']} | {100*r['macro_f1']:.2f} | {delta} | {r['peak_process_mib']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
