"""Analytic hat, complex phase, moment, zero-norm and Conv1d boundary checks."""
import numpy as np
import torch
from torch.nn import functional as F
from radonbridge.fourier_diagnostics import reference_kernels,slice_errors,kernel_gain


def run():
    torch.set_num_threads(3);torch.manual_seed(937)
    for shape in ((5,7),(5,7,5)):
        k=reference_kernels(shape,1,2001);x=np.zeros(shape);center=tuple(n//2 for n in shape);x[center]=1
        # Analytic projection of a tensor-product hat along the first coordinate.
        z=np.maximum(0,1-np.abs(k['s'])/k['h'][0])*k['h'][1:].prod()
        values,profile=slice_errors(x.reshape(1,1,-1),z.reshape(1,1,1,-1),k)
        assert values['full_band_nrmse'][0]<2e-5
        assert values['zeroth_moment_l1_normalized'][0]<2e-5
        zero,_=slice_errors(np.zeros((1,1,np.prod(shape))),np.zeros((1,1,1,2001)),k)
        assert np.isnan(zero['full_band_nrmse']).all()
    # Full cross correlation has a phase-defined frequency response; same is a crop.
    for kernel in (1,3,5):
        weight=torch.randn(3,2,kernel,dtype=torch.float64);x=torch.randn(1,2,11,dtype=torch.float64)
        full=F.conv1d(x,weight,padding=kernel-1)[0].numpy();same=F.conv1d(x,weight,padding=kernel//2)[0].numpy();p=kernel//2
        np.testing.assert_array_equal(full[:,p:p+11],same)
        omega=np.linspace(0,np.pi,129);lags=np.arange(kernel)-p
        h=np.einsum('oik,fk->foi',weight.numpy(),np.exp(1j*omega[:,None]*lags))
        xf=np.einsum('it,ft->fi',x[0].numpy(),np.exp(-1j*omega[:,None]*np.arange(11)))
        yf=np.einsum('ot,ft->fo',full,np.exp(-1j*omega[:,None]*(np.arange(full.shape[1])-p)))
        np.testing.assert_allclose(yf,np.einsum('foi,fi->fo',h,xf),atol=1e-12,rtol=1e-12)
        got=kernel_gain(weight)
        np.testing.assert_allclose(got['frobenius_gain'],np.sqrt((np.abs(h)**2).sum((1,2))),rtol=1e-12,atol=1e-12)
        assert kernel_gain(torch.zeros_like(weight))['zero_operator']
    print('PASS: 2D/3D continuous multilinear Fourier reference, quadrature and moments, undefined zero input, positive kernel phase, full/same boundary crop, Gram frequency gain')


if __name__=='__main__':run()
