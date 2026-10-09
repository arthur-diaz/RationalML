"""One class identity across encoding, probabilities and post-fit diagnostics."""

from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score, f1_score

from rationalml import AutoML


def check_diagnostics(result, X, y, score, class_label=None):
    options = {} if class_label is None else {"class_label": class_label}
    chosen = result.positive_class if class_label is None else class_label
    actual = (y == chosen).astype(int)
    table = result.ranking_table(n_bins=4, **options)
    assert table["positives"].sum() == actual.sum()
    assert table["count"].sum() == len(y)
    top = result.top_segment(fraction=.25, **options)
    score_column = "score" if class_label is None else f"proba_{result.classes_.tolist().index(class_label)}"
    np.testing.assert_allclose(top[score_column], np.sort(score)[::-1][:len(top)])
    calibration = result.calibration_table(n_bins=4, **options)
    assert (calibration["observed_rate"] * calibration["count"]).sum() == pytest.approx(actual.sum())
    summary = result.calibration_summary(n_bins=4, **options)
    assert summary["brier_score"] == pytest.approx(np.mean((score - actual) ** 2))
    # Core runs still exercise every non-SHAP contract. With the extra installed,
    # the very same fitted result must explain the same selected internal output.
    if find_spec("shap") is not None:
        explanation = result.explain(X.iloc[:3], background=X.iloc[3:13], **options)
        assert explanation.output_names == str(chosen)
        raw = result.best_model.decision_function(X.iloc[:3])
        if class_label is not None:
            raw = raw[:, result.classes_.tolist().index(class_label)]
        np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), raw, atol=1e-8)


@pytest.mark.parametrize("positive, negative", [("churn", "retained"), (0, 1), (False, True)])
def test_v1_binary_positive_class_across_all_outputs(binary_df, positive, negative):
    df = binary_df.copy(deep=True)
    df["target"] = np.where(df["target"] == 1, positive, negative)
    result = AutoML(target="target", positive_class=positive, models="logistic_regression",
                    cv=2, n_trials=1, random_state=42, verbose=0).fit(df)
    assert result.positive_class == positive
    assert result.negative_class == negative
    np.testing.assert_array_equal(result.classes_, [negative, positive])
    np.testing.assert_array_equal(result.best_model.classes_, [0, 1])
    X = df.drop(columns="target").iloc[list(result.test_indices)]
    y = df["target"].iloc[list(result.test_indices)].to_numpy()
    encoded = result.label_encoder.transform(df["target"])
    np.testing.assert_array_equal(encoded, (df["target"] == positive).astype(int))
    probabilities = result.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(result.predict_positive_proba(X), probabilities[:, 1])
    stored = result.test_predictions
    np.testing.assert_allclose(stored["score"], probabilities[:, 1])
    np.testing.assert_array_equal(result.predict(X), stored["y_pred"])
    np.testing.assert_allclose(probabilities[:, 0], 1 - stored["score"])
    for name, function in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        assert result.test_metrics[name] == pytest.approx(function(
            y == positive, stored["y_pred"].to_numpy() == positive, zero_division=0))
    check_diagnostics(result, X, y, probabilities[:, 1])


def test_v1_multiclass_zero_is_explicit_valid_label_across_all_outputs(multiclass_df):
    df = multiclass_df.copy(deep=True)
    df["target"] = df["target"].map({"bronze": 0, "gold": 1, "silver": 2})
    result = AutoML(target="target", task="multiclass", models="logistic_regression",
                    cv=2, n_trials=1, random_state=42, verbose=0).fit(df)
    np.testing.assert_array_equal(result.classes_, [0, 1, 2])
    X = df.drop(columns="target").iloc[list(result.test_indices)]
    y = df["target"].iloc[list(result.test_indices)].to_numpy()
    probabilities = result.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    for position, label in enumerate(result.classes_):
        np.testing.assert_allclose(result.test_predictions[f"proba_{position}"], probabilities[:, position])
    np.testing.assert_array_equal(result.predict(X), result.test_predictions["y_pred"])
    check_diagnostics(result, X, y, probabilities[:, 0], class_label=0)
