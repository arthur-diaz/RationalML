import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def binary_df():
    X, y = make_classification(
        n_samples=100, n_features=6, n_informative=4,
        n_redundant=0, random_state=11,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df
