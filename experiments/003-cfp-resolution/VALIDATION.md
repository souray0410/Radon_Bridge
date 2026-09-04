# Engineering validation

Executed on ws before training, with the pinned MHD submodule and LOOK Python
runtime. Source f9830297e6dfce03c06fb65f32b0d160bfae76e8; execution tag
run/003-source (51a19ee2eb0659e5bb6798aeaa47c784b29487fd) adds only the aggregate
analysis script. Training code is unchanged between these two commits.

- 1D–4D Householder direction mapping and involution: maximum error 2.22e-16.
- Adjoint inner-product identity: maximum absolute error 8.88e-16.
- Forward and adjoint autograd gradcheck: all dimensions pass.
- Independent translated-Gaussian projection checks: relative peak errors
  0.023258 (2D), 0.018213 (3D), below the pre-existing 0.10 tolerance.
- Nonzero-bridge MHD versus native PyTorch gradients: maximum error 0;
  parameter update agrees. The tested joint graph has 30 nodes and 26 edges.
- CFP 224 zero-initialized bridge exactly preserves baseline outputs.
- CFP-only and OCT-only graphs both backpropagate and match native forward;
  the excluded branch contributes no operation or parameter to the graph.
- Core checks took 1.779 seconds on ws CPU.
- All 768 original CFP reconstructions at 96 matched cached pixels exactly.
  224 caches are freshly decoded from source images; all OCT NPZ data unchanged.
- During active training, a second nonblocking flock attempt failed as expected.
  The ws working tree was clean at 51a19ee; remote branches and source tags matched.

These checks establish the exercised numerical and graph paths, not clinical
validity, scalable nD performance, or the statistical benefit of Radon geometry.
