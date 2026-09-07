"""Set physical GPUs for all R&B queue phases without changing trial configs."""
import argparse,json
from scripts.gpu_allocation import DEFAULT_PATH,publish,Allocation
from scripts.run_integer_experiment import devices

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,nargs='*',default=None,help='Physical indices; empty list pauses new dispatch')
    p.add_argument('--show',action='store_true');p.add_argument('--file',default=str(DEFAULT_PATH));a=p.parse_args()
    inventory=devices()
    if a.gpus is not None:
        result=publish(a.file,a.gpus,inventory)
        print(json.dumps(dict(written=result,meaning='Active jobs finish; subsequent jobs use only selected GPUs; [] pauses dispatch')))
    elif not a.show:p.error('Use --show or --gpus followed by zero or more physical GPU indices')
    value,_=Allocation(a.file,sorted(inventory)).refresh(inventory)
    print(json.dumps(dict(allocation=value,inventory=inventory)))

if __name__=='__main__':main()
