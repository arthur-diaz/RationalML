"""Core-only example with local, deterministic data; no dataset download."""

import pandas as pd
from sklearn.datasets import make_classification

from rationalml import AutoML


def run():
    X, y = make_classification(n_samples=80, n_features=4, n_informative=3,
                               n_redundant=0, random_state=7)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    df["target"] = y
    result = AutoML(target="target", positive_class=1, preprocessing=None,
                    models=["logistic_regression"], metric="roc_auc",
                    cv=2, n_trials=1, random_state=42, verbose=0).fit(df)
    print(result.leaderboard)
    print(result.test_metrics)
    print(result.ranking_table())
    print(result.calibration_summary())
    assert result.best_model_name == "logistic_regression"
    assert result.predict(df.drop(columns="target")).shape == (len(df),)
    assert result.ranking_table()["count"].sum() == len(result.test_indices)
    return result


if __name__ == "__main__":
    run()
