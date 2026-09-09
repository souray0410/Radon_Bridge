import sys, tempfile
from pathlib import Path
import torch
from tests.integration.check_fixed_channel_basis import basis_checks, artifact
from tests.integration.check_integer_bridge import topology_check, geometry_check

def test_fixed_basis_and_mhd_node_contract():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        assert basis_checks(tmp)["only_mixer_trainable"]
        refs={key:artifact(tmp,c,key)[0] for key,c in [("network0_stage1",2),("network1_stage2",3)]}
        topology_check([2,3],compression="fixed_svd_channel",basis_files=refs)

def test_independent_quadrature_references():
    for d in (2,3,4):geometry_check(d,"cpu")
