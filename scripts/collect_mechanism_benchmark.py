"""Autonomous collection of aggregate report evidence, followed by local PDF QA."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path

def run(a):
    out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    def status(state,**kw):
        p=out/'collection_status.tmp';p.write_text(json.dumps(dict(state=state,pid=os.getpid(),remote=a.root,**kw),indent=2));p.replace(out/'collection_status.json')
    def remote(name):return json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','ws02','cat',a.root+'/'+name],text=True,stderr=subprocess.DEVNULL,timeout=20))
    with (out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                q=remote('queue_status.json')
                if q['state']=='needs_attention':status('remote_needs_attention',details=q);return
                if q['state']=='complete':
                    r=remote('report/report_status.json');assert r['complete_trials']==r['total_trials']==186 and r['seeds']==[3416,3417,3418]
                    subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=10','ws02:'+a.root+'/report/',str(out)+'/'],check=True,timeout=300)
                    d=json.loads((out/'results.json').read_text());assert len(d['rows'])==186 and not d['test_used']
                    status('collected_pending_pdf_authoring_and_visual_review',complete_trials=186);return
                status('waiting_for_remote_report',queue_state=q['state'])
            except (OSError,ValueError,subprocess.SubprocessError) as exc:status('retrying',error=str(exc))
            time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--destination',required=True);run(p.parse_args())
