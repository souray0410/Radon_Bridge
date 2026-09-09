import numpy as np
from sklearn.metrics import f1_score
from radon_bridge.evaluation.metrics import batched_macro_f1

def test_batched_f1_matches_fixed_label_reference():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([[0, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    expected = [f1_score(y, row, average="macro", labels=[0, 1], zero_division=0) for row in pred]
    np.testing.assert_allclose(batched_macro_f1(y, pred), expected, rtol=0, atol=1e-15)
    assert batched_macro_f1(np.zeros(3), np.zeros(3)) == .5
