import json,tempfile
from pathlib import Path
import torch
from radonbridge.svd_basis import save_random_basis,FixedChannelBasis,QR_VERSION,_load_basis
from radonbridge.projector import Projector,LinearResampleProjector,ScrambledProjector
from radonbridge.bridge import FeatureSpec,BridgeExchange
from check_fixed_channel_basis import artifact
from check_integer_bridge import topology_check
from radonbridge.diagnostics import cosine

torch.set_num_threads(3)
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);refs={k:artifact(root,c,k)[0] for k,c in [('network0_stage1',2),('network1_stage2',3)]}
 before=torch.random.get_rng_state().clone();qr={k:save_random_basis(v,root/'qr') for k,v in refs.items()};assert torch.equal(before,torch.random.get_rng_state())
 for key,ref in qr.items():
  a,energy,meta=_load_basis(ref['path'],ref['sha256']);again=save_random_basis(refs[key],root/'again');b,_,_=_load_basis(again['path'],again['sha256'])
  assert torch.equal(a,b) and torch.allclose(a.T@a,torch.eye(len(a),dtype=a.dtype),atol=1e-12,rtol=0)
  full=FixedChannelBasis(len(a),len(a),key,ref,QR_VERSION);assert abs(full.metadata['retained_energy_ratio']-1)<1e-12
 results=[topology_check([2,3],compression='fixed_svd_channel',basis_files=refs,mode=mode) for mode in ['self','scrambled','linear_resample']]
 results.append(topology_check([2,3],compression='fixed_random_orthogonal_channel',basis_files=qr))
 for shape in [(4,5),(3,4,5)]:
  p=Projector(shape,4,11);a=LinearResampleProjector(shape,4,11);n=torch.tensor(shape).prod().item();t=44
  reference=p.backproject(torch.eye(t,dtype=torch.float64).reshape(t,4,11)).reshape(t,n).T
  assert torch.allclose(a.matrix.norm(dim=1),p.matrix.norm(dim=1),atol=1e-12,rtol=1e-12)
  assert torch.allclose(a.return_matrix.norm(dim=1),reference.norm(dim=1),atol=1e-12,rtol=1e-12)
  raw=torch.nn.functional.interpolate(torch.eye(n,dtype=torch.float64)[None],size=t,mode='linear',align_corners=False)[0].T
  assert torch.equal(a.matrix!=0, (raw!=0)&(p.matrix.norm(dim=1,keepdim=True)>0))
  q=ScrambledProjector(shape,4,11,seed=15);assert torch.equal(q.permutation[q.inverse_permutation],torch.arange(n))
 assert cosine(torch.zeros(3),torch.ones(3))['cosine'] is None
 assert cosine(torch.tensor([1.,0.]),torch.tensor([0.,1.]))['cosine']==0
 print(json.dumps({'passed':True,'topology':results,'random_qr_reproducible_rng_preserved':True,'resample_forward_return_row_norms':True,'zero_norm_is_undefined':True}))
