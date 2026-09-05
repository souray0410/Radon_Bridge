"""Copy completed evidence locally; never mutate remote code or training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def run(args):
    out=Path(args.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    def status(state,**kw):
        temp=out/'collection_status.tmp';temp.write_text(json.dumps({'state':state,'pid':os.getpid(),'remote':args.root,**kw},indent=2));temp.replace(out/'collection_status.json')
    def remote(name):return json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','ws02','cat',args.root+'/'+name],text=True,stderr=subprocess.DEVNULL,timeout=20))
    with (out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                queue=remote('queue_status.json')
                if queue['state']=='needs_attention':status('remote_needs_attention',details=queue);return
                if queue['state']=='complete':
                    report=remote('report/report_status.json');assert report['complete_trials']==report['total_trials']==57 and report['seeds']==[3416,3417,3418]
                    subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=10','ws02:'+args.root+'/report/',str(out)+'/'],check=True,timeout=180)
                    data=json.loads((out/'results.json').read_text());assert data['project']=='R&B (Radon Bridge)' and data['test_used'] is False
                    status('collected_pending_visual_review',complete_trials=57);return
                status('waiting_for_remote_report',queue_state=queue['state'])
            except (OSError,ValueError,subprocess.SubprocessError) as exc:status('retrying',error=str(exc))
            time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--destination',required=True);run(p.parse_args())
