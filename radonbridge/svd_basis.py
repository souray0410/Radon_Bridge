"""Training-only, uncentered channel SVD bases, fitted once and then frozen."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import os
import time
import numpy as np
import torch
from torch import nn

BASIS_VERSION='train_channel_second_moment_eigh_v1'


def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_basis(moment, count, directory, source_key, seed, provenance):
    """Eigenvectors of FF^T/N are the uncentered left singular vectors of F."""
    if moment.dtype!=torch.float64 or moment.device.type!='cpu' or moment.ndim!=2 or moment.shape[0]!=moment.shape[1] or count<=0 or not torch.isfinite(moment).all():
        raise ValueError('Expected finite CPU float64 channel second moment and positive count')
    covariance=(moment+moment.T)/(2*count)
    values,q=torch.linalg.eigh(covariance)
    if values[0]<-1e-10*max(1.,float(values[-1])) or values.sum()<=0:
        raise ValueError('Invalid or zero-energy second moment')
    values=values.flip(0).clamp_min(0);q=q.flip(1)
    pivots=q.abs().argmax(dim=0)
    signs=torch.where(q[pivots,torch.arange(q.shape[1])]<0,-1.,1.)
    q=(q*signs).contiguous()
    metadata={'version':BASIS_VERSION,'source_key':source_key,'seed':seed,'channels':q.shape[0],
              'sampled_channel_vectors':count,'centered':False,'fit_split':'train','test_used':False,
              'normalization':'FF^T/N; energy values are squared singular values divided by N',
              'ordering':'descending energy; maximum-absolute entry of each column nonnegative',
              'full_master_sha256':tensor_sha(q),'torch_version':torch.__version__,'provenance':provenance}
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps([seed,source_key,BASIS_VERSION],separators=(',',':')).encode()).hexdigest()[:16]
    path=directory/f'seed{seed}_{identity}.npz'
    if path.exists():raise RuntimeError(f'Refuse to overwrite fitted basis {path}')
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez(f,q=q.numpy(),eigenvalues=values.numpy(),second_moment=covariance.numpy(),metadata=json.dumps(metadata,sort_keys=True))
    tmp.replace(path)
    return {'path':str(path.resolve()),'sha256':file_sha(path)}


@lru_cache(maxsize=64)
def _load_basis(path, expected_sha):
    if file_sha(path)!=expected_sha:raise ValueError('SVD basis file SHA256 mismatch')
    with np.load(path,allow_pickle=False) as z:
        q=torch.from_numpy(z['q'].copy());values=torch.from_numpy(z['eigenvalues'].copy());metadata=json.loads(str(z['metadata']))
    if metadata['version']!=BASIS_VERSION or metadata['fit_split']!='train' or metadata['centered'] or metadata['test_used']:
        raise ValueError('Basis must be the accepted training-only uncentered SVD')
    if q.dtype!=torch.float64 or q.shape!=(metadata['channels'],metadata['channels']) or not torch.isfinite(q).all() or tensor_sha(q)!=metadata['full_master_sha256']:
        raise ValueError('Invalid SVD basis tensor')
    if not torch.allclose(q.T@q,torch.eye(len(q),dtype=q.dtype),atol=1e-10,rtol=0):raise ValueError('Basis is not orthogonal')
    if values.shape!=(len(q),) or not torch.isfinite(values).all() or (values<0).any() or (values[1:]>values[:-1]).any() or values.sum()<=0:
        raise ValueError('Invalid ordered energy spectrum')
    return q,values,metadata


class FixedChannelBasis(nn.Module):
    def __init__(self, channels, retained, source_key, artifact):
        super().__init__()
        if not isinstance(artifact,dict) or set(artifact)!={'path','sha256'} or not 1<=retained<=channels:
            raise ValueError('Fixed SVD compression requires a path/SHA256 basis reference')
        full,values,metadata=_load_basis(artifact['path'],artifact['sha256'])
        if metadata['source_key']!=source_key or metadata['channels']!=channels:
            raise ValueError('SVD basis source or channel count mismatch')
        self.artifact=dict(artifact);self._master=full[:,:retained].clone()
        self.register_buffer('q',self._master.clone())
        self.metadata=dict(metadata,channel_rank=retained,retained_energy_ratio=float(values[:retained].sum()/values.sum()),
                           retained_master_sha256=tensor_sha(self._master),artifact=self.artifact)

    def encode(self,x):return torch.einsum('cr,bc...->br...',self.q,x)
    def decode(self,x):return torch.einsum('cr,br...->bc...',self.q,x)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        saved=state_dict.get(prefix+'q')
        if saved is not None and (saved.shape!=self._master.shape or not torch.equal(saved.cpu(),self._master.to(dtype=saved.dtype))):
            error_msgs.append('Checkpoint basis does not match its accepted SVD artifact')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def export(self,directory):
        # Preserve the authoritative full basis; save the actual runtime buffer separately.
        if file_sha(self.artifact['path'])!=self.artifact['sha256']:raise ValueError('SVD artifact changed')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);digest=tensor_sha(self.q);path=directory/(digest+'.npy')
        if not path.exists():
            tmp=path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:np.save(f,self.q.detach().cpu().numpy(),allow_pickle=False)
            tmp.replace(path)
        if hashlib.sha256(np.load(path,allow_pickle=False).tobytes()).hexdigest()!=digest:raise ValueError('Runtime basis artifact mismatch')
        return dict(self.metadata,runtime_artifact={'path':str(path.resolve()),'tensor_sha256':digest,'file_sha256':file_sha(path),'dtype':str(self.q.dtype)})


def fit_training_bases(config, output, data):
    from .experiment import parameter_hash,loader,write_json
    from .data import PairedDataset
    from .model import PilotGraph
    start=time.monotonic();torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(9*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():raise RuntimeError('Basis fit already completed')
    train=PairedDataset(data,'train',224);assert len(train)==1264
    seed=config['seed'];g=PilotGraph(seed=seed,device='cuda');g.graph.eval()
    for branch,parent in config['parent_checkpoints'].items():
        assert file_sha(parent['path'])==parent['sha256']
        saved=torch.load(parent['path'],map_location='cpu',weights_only=False)
        assert saved['branch']==branch and saved['seed']==seed and saved['training_stage']=='independent' and saved['stop_reason']=='validation_plateau'
        g.load_native_state(saved['model'],branch)
    assert set(config['parent_checkpoints'])=={'cfp','oct'}
    initial=parameter_hash(g);moments={};counts={};ids=[]
    with torch.no_grad():
        for c,o,y,keys in loader(train,config['microbatch'],seed):
            g.forward(c.cuda(),o.cuda(),y.cuda());ids.extend(keys)
            for key in config['nodes']:
                x=g.by_name[key].feature_message.current_state.detach()
                f=x.movedim(1,0).reshape(x.shape[1],-1).cpu().double()
                moments[key]=moments.get(key,torch.zeros(f.shape[0],f.shape[0],dtype=torch.float64))+f@f.T
                counts[key]=counts.get(key,0)+f.shape[1]
            write_json(output/'progress.json',{'participants':len(ids),'seconds':time.monotonic()-start})
    assert len(ids)==1264 and len(set(ids))==1264 and parameter_hash(g)==initial
    provenance={'parent_checkpoints':config['parent_checkpoints'],'initial_native_sha256':initial,
                'participant_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                'participants':len(ids),'source_commit':config['source_commit'],'data_audit_sha256':file_sha(Path(data)/'audit.json'),
                'batchnorm':'eval; unchanged','fit_domain':'native stage3 channel features; all eyes and spatial positions equally weighted'}
    bases={key:save_basis(moment,counts[key],output/'bases',key,seed,provenance) for key,moment in moments.items()}
    write_json(output/'summary.json',{'state':'complete','passed':True,'seed':seed,'bases':bases,'provenance':provenance,'seconds':time.monotonic()-start,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'test_used':False})

if __name__=='__main__':
    import argparse,traceback
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--data',required=True);a=parser.parse_args()
    try:fit_training_bases(json.loads(Path(a.config).read_text()),a.output,a.data)
    except Exception as exc:
        from .experiment import write_json
        write_json(Path(a.output)/'failure.json',{'state':'oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'failed','error':traceback.format_exc()});raise
