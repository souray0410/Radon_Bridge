import numpy as np
import pytest

from radon_bridge.evaluation.metrics import binary_metrics


def test_binary_metrics_preserve_native_schema_and_probability_scores():
    labels = [0, 0, 1, 1]
    probabilities = [[.9, .1], [.4, .6], [.2, .8], [.7, .3]]
    result = binary_metrics(labels, probabilities)
    assert set(result) == {
        "macro_f1", "auroc", "positive_average_precision", "nll", "brier", "confusion_matrix"
    }
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["macro_f1"] == .5
    assert result["auroc"] == .75
    assert result["positive_average_precision"] == pytest.approx(5 / 6)
    assert result["brier"] == pytest.approx(.225)
    assert result["nll"] == pytest.approx(-np.log([.9, .4, .8, .3]).mean())
