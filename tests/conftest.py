import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def binary_df():
    X, y = make_classification(
        n_samples=100, n_features=6, n_informative=4,
        n_redundant=0, random_state=11,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def multiclass_df():
    X, y = make_classification(
        n_samples=150, n_features=6, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=23,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = np.array(["silver", "bronze", "gold"])[y]
    return df


@pytest.fixture
def regression_df():
    X, y = make_regression(n_samples=120, n_features=6, noise=4, random_state=29)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(31)
    count = 120
    y = np.arange(count) % 2
    df = pd.DataFrame({
        "numeric_float": rng.normal(size=count) + 2 * y,
        "numeric_int": rng.integers(18, 80, size=count),
        "boolean": pd.Series(rng.choice([True, False], count), dtype="boolean"),
        "categorical": pd.Series(rng.choice(["France", "Germany", "Spain"], count), dtype=object),
        "category_dtype": pd.Categorical(rng.choice(["a", "b"], count), categories=["a", "b", "unused"]),
        "string_dtype": pd.Series(rng.choice(["alpha", "beta"], count), dtype="string"),
        "nullable_int": pd.Series(rng.integers(1, 40, size=count), dtype="Int64"),
        "target": np.where(y == 1, "churn", "retained"),
    })
    for name, interval in [("numeric_float", 9), ("boolean", 11), ("categorical", 13),
                           ("category_dtype", 17), ("string_dtype", 19), ("nullable_int", 23)]:
        df.loc[::interval, name] = None if name == "categorical" else pd.NA
    return df
