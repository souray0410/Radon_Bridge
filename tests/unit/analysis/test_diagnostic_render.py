import json
import numpy as np
from radon_bridge.analysis.native_diagnostics import render_sinograms
from radon_bridge.runtime.state import file_sha256


def sample(path,value=1):
    np.savez(path,projection_rms=np.full((3,4),value),support=np.array([-1,1]),directions=np.eye(3))


def test_extension_preserves_accepted_images(tmp_path):
    sample(tmp_path/'core_sinogram.npz')
    render_sinograms(tmp_path)
    previous={p.name:file_sha256(p) for p in tmp_path.iterdir()}
    sample(tmp_path/'extra_sinogram.npz',2)
    render_sinograms(tmp_path)
    assert all(file_sha256(tmp_path/name)==sha for name,sha in previous.items())
    assert (tmp_path/'extra_sinogram.svg').exists()


def test_changed_source_and_damaged_output_are_not_reused(tmp_path):
    source=tmp_path/'core_sinogram.npz';sample(source)
    render_sinograms(tmp_path)
    sample(source,2);render_sinograms(tmp_path)
    receipt=source.with_suffix('.render.json')
    assert json.loads(receipt.read_text())['source_sha256']==file_sha256(source)
    source.with_suffix('.png').write_bytes(b'damaged')
    render_sinograms(tmp_path)
    saved=json.loads(receipt.read_text())
    assert saved['files'][source.with_suffix('.png').name]==file_sha256(source.with_suffix('.png'))
    assert source.with_suffix('.png').read_bytes()!=b'damaged'
