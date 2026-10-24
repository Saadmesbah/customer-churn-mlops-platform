import numpy as np
import pytest

from churn.train import metrics


def test_known_metrics():
    result = metrics([0, 0, 1, 1], np.array([0.1, 0.8, 0.7, 0.9]))
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
