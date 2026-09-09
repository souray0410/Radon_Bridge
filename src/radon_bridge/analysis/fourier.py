"""Bounded CPU-only Radon/Fourier diagnostics, separate from model selection."""
import numpy as np
import torch
from radon_bridge.methods.projection import geometry, eem_directions


def reference_kernels(shape,M,S,points=33):
    h,radius,s,_=geometry(shape,S);n,_=eem_directions(len(shape),M)
    coords=np.stack(np.meshgrid(*[(np.arange(v)-(v-1)/2)*step for v,step in zip(shape,h)],indexing='ij'),-1).reshape(-1,len(shape))
    maxfreq=min(.5/(s[1]-s[0]),.5/h.max());freq=np.linspace(0,maxfreq,points)
    xi=n[:,None,:]*freq[None,:,None]
    kernel=np.exp(-2j*np.pi*np.einsum('mfd,nd->mfn',xi,coords))
    kernel*=np.prod(h*np.sinc(xi*h)**2,axis=-1)[...,None]
    quadrature=np.ones(S)*(s[1]-s[0]);quadrature[[0,-1]]*=.5
    projection=np.exp(-2j*np.pi*freq[:,None]*s)*quadrature
    return dict(spatial=kernel,projection=projection,frequencies=freq,coords=coords,h=h,s=s,n=n,radius=radius,
                frequency_fraction=freq/maxfreq,native_nyquist=.5/h.max(),projection_nyquist=.5/(s[1]-s[0]))


def slice_errors(x,z,k):
    # x [participant, eye*rank, N]; z [participant, eye*rank, M, S].
    expected=np.einsum('bcn,mfn->bcmf',x,k['spatial'],optimize=True)
    actual=np.einsum('bcms,fs->bcmf',z,k['projection'],optimize=True)
    error=np.abs(actual-expected)**2;reference=np.abs(expected)**2
    ratio=lambda a,b:np.sqrt(np.divide(a,b,out=np.full_like(a,np.nan),where=b>0))
    low=k['frequency_fraction']<=.5
    result=dict(full_band_nrmse=ratio(error.sum((1,2,3)),reference.sum((1,2,3))),
        low_band_nrmse=ratio(error[...,low].sum((1,2,3)),reference[...,low].sum((1,2,3))),
        absolute_rmse=np.sqrt(error.mean((1,2,3))),reference_rms=np.sqrt(reference.mean((1,2,3))))
    # Signed features may have zero DC; normalize moments by an L1 upper bound, not signed mass.
    volume=float(k['h'].prod());q=np.ones(len(k['s']))*(k['s'][1]-k['s'][0]);q[[0,-1]]*=.5
    m0=np.einsum('bcms,s->bcm',z,q);true0=x.sum(-1)*volume
    m1=np.einsum('bcms,s->bcm',z,q*k['s']);true1=np.einsum('bcn,nd,md->bcm',x,k['coords'],k['n'])*volume
    bound=np.abs(x).sum(-1)*volume
    d0=np.mean(bound**2,axis=1);d1=d0*k['radius']**2
    result['zeroth_moment_l1_normalized']=ratio(((m0-true0[...,None])**2).mean((1,2)),d0)
    result['first_moment_l1_radius_normalized']=ratio(((m1-true1)**2).mean((1,2)),d1)
    profile=dict(error_squared=error.mean((0,1,2)).tolist(),reference_squared=reference.mean((0,1,2)).tolist(),
        frequencies=k['frequencies'].tolist(),frequency_fraction=k['frequency_fraction'].tolist(),
        cycles_per_s_sample=(k['frequencies']*(k['s'][1]-k['s'][0])).tolist(),native_nyquist=k['native_nyquist'],projection_nyquist=k['projection_nyquist'])
    return result,profile


def summarize(values):
    out={}
    for key,v in values.items():
        v=np.asarray(v);good=v[np.isfinite(v)]
        out[key]=dict(mean=float(good.mean()) if len(good) else None,sample_sd=float(good.std(ddof=1)) if len(good)>1 else None,
            defined=len(good),undefined=len(v)-len(good))
    return out


def kernel_gain(weight,points=129):
    # Stored Conv1d is cross correlation. Frobenius norm is computed from lag Gram.
    w=weight.detach().cpu().double();k=w.shape[-1];lags=np.arange(k)-k//2;flat=w.reshape(-1,k)
    gram=(flat.T@flat).numpy();omega=np.linspace(0,np.pi,points)
    power=np.einsum('ij,fij->f',gram,np.cos(omega[:,None,None]*(lags[:,None]-lags[None,:])),optimize=True)
    power=np.maximum(power,0)
    return dict(cycles_per_sample=(omega/(2*np.pi)).tolist(),frobenius_gain=np.sqrt(power).tolist(),
        rms_per_channel_pair_gain=np.sqrt(power/(w.shape[0]*w.shape[1])).tolist(),lag_gram=gram.tolist(),
        stored_shape=list(w.shape),zero_operator=bool(torch.count_nonzero(w)==0))
