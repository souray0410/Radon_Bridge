import numpy as np
import pytest
import torch

from radon_bridge.runtime.next_update_replay import compare_state, validate_sbatch_header


def test_compare_state_handles_numpy_rng_without_ambiguous_truth():
    value = {"rng": ("MT19937", np.arange(624, dtype=np.uint32)), "tensor": torch.ones(2)}
    compare_state(value, {"rng": ("MT19937", value["rng"][1].copy()), "tensor": torch.ones(2)})
    changed = value["rng"][1].copy();changed[4] += 1
    with pytest.raises(ValueError, match="numpy value"):
        compare_state(value, {"rng": ("MT19937", changed), "tensor": torch.ones(2)})


def test_sbatch_directives_must_precede_shell_commands():
    assert validate_sbatch_header("#!/bin/bash\n#SBATCH --output=x\nset -e\n")
    with pytest.raises(ValueError, match="follows"):
        validate_sbatch_header("#!/bin/bash\nset -e\n#SBATCH --output=x\n")
