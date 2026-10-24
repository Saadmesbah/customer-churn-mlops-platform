import pytest

from churn.data import FEATURES


@pytest.fixture
def customer():
    values = [2, 0, 30, 3, 4000, 80, 10, 20, 3, 1, 1, 35, 100]
    return dict(zip(FEATURES, values))
