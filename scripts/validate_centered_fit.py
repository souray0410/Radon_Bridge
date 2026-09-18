"""Independent numeric acceptance for immutable WS02 centered-SVD fit artifacts.

This script intentionally does not call cohort_centered.validate_basis_fit or the
basis-saving functions. It recomputes the centered definition from the saved
sufficient statistics and fails closed before formal training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA='radon_centered_numeric_acceptance_v1'
ATOL_ARRAY=1e-12
RTOL_ARRAY=1e-12
ORTH_TOL=1e-10
EIGEN_RESIDUAL_TOL=1e-10
SORT_TOL=1e-12
SYMMETRY_TOL=1e-12


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def _metadata(path):
    with np.load(path,allow_pickle=False) as z:
        return json.loads(str(z['metadata']))


def _same_parent_sha(old,new):
    return (isinstance(old,dict) and isinstance(new,dict) and set(old)==set(new)
            and all(old[k].get('sha256')==new[k].get('sha256') for k in old))


def recompute_fingerprint(config,data,initial_native_sha256):
    data=Path(data)
    parents={k:v['sha256'] for k,v in sorted(config['parent_checkpoints'].items())}
    uncentered={k:v['sha256'] for k,v in sorted(config.get('uncentered_basis_files',{}).items())}
    payload=dict(schema='train_channel_sufficient_statistics_v1',seed=config['seed'],parents=parents,
        nodes=list(config['nodes']),microbatch=config['microbatch'],source_commit=config['source_commit'],
        fit_centered=bool(config.get('fit_centered',False)),uncentered_basis_sha256=uncentered,
        data_audit_sha256=file_sha(data/'audit.json'),selected_sha256=file_sha(data/'selected.csv'),
        preprocessing='PairedDataset train, CFP224 normalization, OCT fixed normalization',
        sampling='all eyes and spatial positions; deterministic train order; no shuffle',
        batchnorm='eval; unchanged',fit_domain=config.get('fit_domain','native stage3 channel features; all eyes and spatial positions equally weighted'),
        initial_native_sha256=initial_native_sha256,
        basis_versions=['train_channel_second_moment_eigh_v1','train_channel_centered_covariance_eigh_v1'])
    text=json.dumps(payload,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest(),payload


def _assert_close(name,actual,expected,atol=ATOL_ARRAY,rtol=RTOL_ARRAY):
    if not np.allclose(actual,expected,atol=atol,rtol=rtol):
        error=float(np.max(np.abs(actual-expected)))
        raise ValueError(f'{name} mismatch; max_abs={error}')


def validate(root):
    root=Path(root)
    config=json.loads((root/'basis_fit_config.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'basis_fit/summary.json').read_text())
    state_path=root/'basis_fit/sufficient_statistics.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=False)

    if (protocol.get('study_kind')!='centered_basis' or protocol.get('test_used') is not False
            or config.get('fit_centered') is not True or config.get('seed')!=3416
            or config.get('nodes')!=['cfp_stage3','oct_stage3'] or config.get('microbatch')!=16):
        raise ValueError('Centered numeric protocol identity changed')
    if summary.get('state')!='complete' or summary.get('passed') is not True or summary.get('test_used') is not False:
        raise ValueError('Centered fit is not a complete training-only fit')
    if state.get('schema')!='train_channel_sufficient_statistics_state_v1':
        raise ValueError('Unknown centered sufficient-statistics schema')
    if state.get('offset')!=1264 or len(state.get('ids',[]))!=1264 or len(set(state['ids']))!=1264:
        raise ValueError('Centered sufficient-statistics participant coverage changed')
    if set(state.get('moments',{}))!=set(config['nodes']) or set(state.get('sums',{}))!=set(config['nodes']) or set(state.get('counts',{}))!=set(config['nodes']):
        raise ValueError('Centered sufficient-statistics source set changed')

    old_meta={}
    initial_native=None
    old_ids_sha=None
    for key in config['nodes']:
        ref=config['uncentered_basis_files'][key]
        if file_sha(ref['path'])!=ref['sha256']:
            raise ValueError('Accepted uncentered basis bytes changed')
        meta=_metadata(ref['path'])
        if (meta.get('version')!='train_channel_second_moment_eigh_v1' or meta.get('centered') is not False
                or meta.get('source_key')!=key or meta.get('seed')!=3416 or meta.get('test_used') is not False):
            raise ValueError('Accepted uncentered basis metadata changed')
        provenance=meta.get('provenance',{})
        if not _same_parent_sha(provenance.get('parent_checkpoints',{}),config['parent_checkpoints']):
            raise ValueError('Accepted uncentered parent identity changed')
        candidate_native=provenance.get('initial_native_sha256')
        candidate_ids=provenance.get('participant_ids_sha256')
        if not candidate_native or not candidate_ids:
            raise ValueError('Accepted uncentered basis lacks native/participant identity')
        initial_native=candidate_native if initial_native is None else initial_native
        old_ids_sha=candidate_ids if old_ids_sha is None else old_ids_sha
        if candidate_native!=initial_native or candidate_ids!=old_ids_sha:
            raise ValueError('Accepted uncentered basis provenance disagrees across sources')
        old_meta[key]=meta

    actual_ids_sha=ids_sha(state['ids'])
    if actual_ids_sha!=old_ids_sha or actual_ids_sha!=summary.get('provenance',{}).get('participant_ids_sha256'):
        raise ValueError('Centered sufficient-statistics participant order mismatch')

    fingerprint,payload=recompute_fingerprint(config,protocol['data'],initial_native)
    summary_provenance=summary.get('provenance',{})
    if (state.get('fingerprint')!=fingerprint or summary_provenance.get('statistics_fingerprint')!=fingerprint
            or summary_provenance.get('statistics_fingerprint_payload')!=payload):
        raise ValueError('Centered sufficient-statistics fingerprint mismatch')
    if summary_provenance.get('sufficient_statistics_sha256')!=file_sha(state_path):
        raise ValueError('Centered sufficient-statistics SHA mismatch')
    if (summary_provenance.get('initial_native_sha256')!=initial_native
            or not _same_parent_sha(summary_provenance.get('parent_checkpoints',{}),config['parent_checkpoints'])):
        raise ValueError('Centered fit native/parent identity mismatch')
    if set(summary.get('bases',{}))!=set(config['nodes']):
        raise ValueError('Centered fitted basis source set changed')

    measured={}
    for key in config['nodes']:
        moment=state['moments'][key]
        channel_sum=state['sums'][key]
        count=state['counts'][key]
        if (not isinstance(moment,torch.Tensor) or moment.dtype!=torch.float64 or moment.device.type!='cpu'
                or not isinstance(channel_sum,torch.Tensor) or channel_sum.dtype!=torch.float64 or channel_sum.device.type!='cpu'
                or type(count) is not int or count<=0 or moment.ndim!=2 or moment.shape[0]!=moment.shape[1]
                or channel_sum.shape!=(moment.shape[0],) or not torch.isfinite(moment).all() or not torch.isfinite(channel_sum).all()):
            raise ValueError('Invalid centered sufficient-statistics tensors')
        if count!=old_meta[key].get('sampled_channel_vectors'):
            raise ValueError('Centered sufficient-statistics count differs from accepted uncentered fit')
        m=moment.numpy();s=channel_sum.numpy()
        moment_sym=float(np.max(np.abs(m-m.T)))
        second=m/count
        second_sym=float(np.max(np.abs(second-second.T)))
        if second_sym>SYMMETRY_TOL:
            raise ValueError(f'Centered normalized second moment is not symmetric; max_abs={second_sym}')
        mean=s/count
        covariance=second-np.outer(mean,mean)

        ref=summary['bases'][key]
        if file_sha(ref['path'])!=ref['sha256']:
            raise ValueError('Centered basis bytes changed')
        with np.load(ref['path'],allow_pickle=False) as z:
            required={'q','eigenvalues','energies','mean','second_moment','covariance','metadata'}
            if set(z.files)!=required:
                raise ValueError('Centered basis arrays changed')
            q=z['q'].copy();eigen=z['eigenvalues'].copy();energy=z['energies'].copy()
            saved_mean=z['mean'].copy();saved_second=z['second_moment'].copy();saved_cov=z['covariance'].copy()
            meta=json.loads(str(z['metadata']))

        if (q.dtype!=np.float64 or eigen.dtype!=np.float64 or energy.dtype!=np.float64
                or saved_mean.dtype!=np.float64 or saved_second.dtype!=np.float64 or saved_cov.dtype!=np.float64
                or q.shape!=(moment.shape[0],moment.shape[0]) or eigen.shape!=(moment.shape[0],)
                or energy.shape!=(moment.shape[0],)):
            raise ValueError('Centered basis numeric shape/dtype changed')
        if not all(np.isfinite(v).all() for v in (q,eigen,energy,saved_mean,saved_second,saved_cov)):
            raise ValueError('Centered basis contains non-finite values')
        if (meta.get('version')!='train_channel_centered_covariance_eigh_v1' or meta.get('centered') is not True
                or meta.get('runtime_centering') is not False or meta.get('source_key')!=key
                or meta.get('sampled_channel_vectors')!=count or meta.get('seed')!=3416 or meta.get('test_used') is not False):
            raise ValueError('Centered basis metadata changed')
        if meta.get('provenance',{}).get('statistics_fingerprint')!=fingerprint:
            raise ValueError('Centered basis provenance fingerprint mismatch')
        if meta.get('provenance',{}).get('sufficient_statistics_sha256')!=file_sha(state_path):
            raise ValueError('Centered basis provenance statistics SHA mismatch')
        if tensor_sha(q)!=meta.get('full_master_sha256') or tensor_sha(saved_mean)!=meta.get('mean_sha256'):
            raise ValueError('Centered basis tensor identity metadata mismatch')

        _assert_close(f'{key} mean=sum/N',saved_mean,mean)
        _assert_close(f'{key} second=moment/N',saved_second,second)
        _assert_close(f'{key} covariance=second-mean_outer',saved_cov,covariance)
        cov_sym=float(np.max(np.abs(covariance-covariance.T)))
        if cov_sym>SYMMETRY_TOL:
            raise ValueError(f'{key} covariance not symmetric; max_abs={cov_sym}')

        orth=q.T@q-np.eye(q.shape[1])
        orth_error=float(np.max(np.abs(orth)))
        if orth_error>ORTH_TOL:
            raise ValueError(f'{key} basis not orthogonal; max_abs={orth_error}')
        scale=max(1.0,float(np.max(np.abs(covariance))),float(np.max(np.abs(eigen))))
        eigen_residual=covariance@q-q*eigen[None,:]
        eigen_error=float(np.max(np.abs(eigen_residual)))/scale
        if eigen_error>EIGEN_RESIDUAL_TOL:
            raise ValueError(f'{key} covariance eigen residual too large; relative={eigen_error}')
        sort_violation=float(np.max(np.maximum(eigen[1:]-eigen[:-1],0.0))) if len(eigen)>1 else 0.0
        if sort_violation>SORT_TOL or float(np.min(eigen)) < -SORT_TOL:
            raise ValueError(f'{key} centered eigen spectrum is not sorted/nonnegative')
        pivots=np.argmax(np.abs(q),axis=0)
        if np.any(q[pivots,np.arange(q.shape[1])]<-SORT_TOL):
            raise ValueError(f'{key} centered eigenvector sign convention changed')

        expected_energy=np.sum(q*(second@q),axis=0)
        _assert_close(f'{key} qT-second-q energy',energy,expected_energy)
        measured[key]={
            'count':count,
            'raw_moment_symmetry_max_abs':moment_sym,
            'second_symmetry_max_abs':second_sym,
            'mean_max_abs':float(np.max(np.abs(saved_mean-mean))),
            'second_max_abs':float(np.max(np.abs(saved_second-second))),
            'covariance_max_abs':float(np.max(np.abs(saved_cov-covariance))),
            'orthogonality_max_abs':orth_error,
            'eigen_residual_relative':eigen_error,
            'eigen_sort_violation':sort_violation,
            'energy_max_abs':float(np.max(np.abs(energy-expected_energy))),
            'basis_sha256':ref['sha256'],
        }

    return {
        'schema':SCHEMA,'passed':True,'test_used':False,'participants':1264,
        'participant_ids_sha256':actual_ids_sha,'statistics_fingerprint':fingerprint,
        'sufficient_statistics_sha256':file_sha(state_path),
        'basis_fit_config_sha256':file_sha(root/'basis_fit_config.json'),
        'fit_summary_sha256':file_sha(root/'basis_fit/summary.json'),
        'tolerances':{
            'array_atol':ATOL_ARRAY,'array_rtol':RTOL_ARRAY,'symmetry_abs':SYMMETRY_TOL,
            'orthogonality_abs':ORTH_TOL,'eigen_residual_relative':EIGEN_RESIDUAL_TOL,'sort_abs':SORT_TOL,
        },
        'measured':measured,
    }


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--validator-commit',required=True)
    args=p.parse_args();value=validate(args.root)
    value['validator_commit']=args.validator_commit
    value['validator_script_sha256']=file_sha(__file__)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        current=json.loads(out.read_text())
        if current!=value:raise ValueError('Centered numeric acceptance receipt changed')
    else:
        tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(out)
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
