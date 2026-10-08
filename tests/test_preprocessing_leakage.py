import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split

from rationalml import AutoML
from rationalml.exceptions import DataValidationError


def leakage_dataset():
    """Each validation block has a separate numeric range and category."""
    df = pd.DataFrame({"amount": np.zeros(60), "country": ["unset"] * 60, "target": np.arange(60) % 2})
    train_rows, test_rows = train_test_split(
        np.arange(len(df)), stratify=df.target, test_size=0.2, random_state=17,
    )
    train = df.iloc[train_rows]
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=17).split(train, train.target))
    for number, (_, validation_rows) in enumerate(folds):
        rows = train_rows[validation_rows]
        df.loc[rows, "amount"] = [1.0, 1_001.0, 100_001.0][number]
        df.loc[rows[0], "amount"] = np.nan
        df.loc[rows, "country"] = f"fold_{number}"
    df.loc[test_rows, "amount"] = 1_000_000_000.0
    df.loc[test_rows, "country"] = "holdout_only"
    # Unobserved categories in pandas metadata must not enter the fitted OHE.
    df.country = pd.Categorical(df.country, categories=["fold_0", "fold_1", "fold_2", "holdout_only", "unused"])
    return df, train_rows, test_rows, folds


def test_every_fold_fits_its_own_schema_imputers_encoder_and_scaler_on_fold_train(monkeypatch):
    import rationalml.preprocessing.builder as builder

    df, train_rows, test_rows, folds = leakage_dataset()
    original = df.copy(deep=True)
    fit_events, transform_events, imputer_events, schema_events = [], [], [], []
    real_fit_transform = ColumnTransformer.fit_transform
    real_transform = ColumnTransformer.transform
    real_imputer_fit = SimpleImputer.fit
    real_infer_schema = builder.infer_schema

    def record_fit(self, X, y=None, **options):
        result = real_fit_transform(self, X, y, **options)
        numeric = self.named_transformers_["numeric"].named_steps
        categorical = self.named_transformers_["categorical"].named_steps
        fit_events.append({
            "preprocessor": self, "rows": tuple(X.index),
            "numeric_imputer": numeric["imputer"], "categorical_imputer": categorical["imputer"],
            "encoder": categorical["encoder"], "scaler": numeric["scaler"],
            "median": numeric["imputer"].statistics_[0], "mean": numeric["scaler"].mean_[0],
            "categories": tuple(categorical["encoder"].categories_[0]),
        })
        return result

    def record_transform(self, X, **options):
        transform_events.append((self, set(X.index)))
        return real_transform(self, X, **options)

    def record_imputer_fit(self, X, y=None):
        imputer_events.append((self, set(X.index)))
        return real_imputer_fit(self, X, y)

    def record_schema(X):
        schema_events.append(set(X.index))
        return real_infer_schema(X)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", record_fit)
    monkeypatch.setattr(ColumnTransformer, "transform", record_transform)
    monkeypatch.setattr(SimpleImputer, "fit", record_imputer_fit)
    monkeypatch.setattr(builder, "infer_schema", record_schema)
    result = AutoML(target="target", models="logistic_regression", preprocessing="basic", cv=3, n_trials=2,
                    random_state=17, verbose=0).fit(df)
    assert result.train_indices == tuple(train_rows)
    assert result.test_indices == tuple(test_rows)
    train, holdout = set(train_rows), set(test_rows)
    full_median = df.iloc[train_rows].amount.median()
    assert full_median == 1_001
    assert len(fit_events) == 2 * 3 + 1
    assert len(imputer_events) == 2 * len(fit_events)
    assert schema_events == [set(event["rows"]) for event in fit_events]
    for key in ["preprocessor", "numeric_imputer", "categorical_imputer", "encoder", "scaler"]:
        assert len({id(event[key]) for event in fit_events}) == len(fit_events)
    for event in fit_events:
        rows = set(event["rows"])
        assert rows <= train and rows.isdisjoint(holdout)
        for key in ["numeric_imputer", "categorical_imputer"]:
            assert [seen for imputer, seen in imputer_events if imputer is event[key]] == [rows]
    # CV fits appear in trial/fold order. Each learns its own median and category set.
    for number, event in enumerate(fit_events[:-1]):
        fold_train, fold_validation = folds[number % 3]
        expected_train = set(train_rows[fold_train])
        expected_validation = set(train_rows[fold_validation])
        assert set(event["rows"]) == expected_train
        assert expected_train.isdisjoint(expected_validation)
        fold_values = df.iloc[train_rows[fold_train]].amount
        assert event["median"] == pytest.approx(fold_values.median())
        assert abs(event["median"] - full_median) >= 500
        assert event["mean"] == pytest.approx(fold_values.fillna(fold_values.median()).mean())
        assert set(event["categories"]) == set(df.iloc[train_rows[fold_train]].country)
        assert f"fold_{number % 3}" not in event["categories"]
        assert [seen for preprocessor, seen in transform_events if preprocessor is event["preprocessor"]] == [
            expected_validation,
        ]
    final = fit_events[-1]
    assert set(final["rows"]) == train
    assert final["median"] == full_median
    assert set(final["categories"]) == {"fold_0", "fold_1", "fold_2"}
    assert final["preprocessor"] is result.best_model.named_steps["preprocessing"]
    # One final evaluation: one predict and one predict_proba transform, no test fit.
    assert [seen for preprocessor, seen in transform_events if preprocessor is final["preprocessor"]] == [holdout, holdout]
    assert all(seen <= train or seen == holdout for _, seen in transform_events)
    pd.testing.assert_frame_equal(df, original)


def test_changes_to_holdout_cannot_change_schema_statistics_or_model_selection(mixed_df):
    options = dict(target="target", positive_class="churn", models="auto", preprocessing="basic", cv=3,
                   n_trials=2, random_state=17, verbose=0)
    first = AutoML(**options).fit(mixed_df)
    changed = mixed_df.copy(deep=True)
    heldout = list(first.test_indices)
    changed.loc[heldout, "numeric_float"] = 1e12
    changed.loc[heldout, "nullable_int"] = pd.NA
    changed.loc[heldout, "categorical"] = "holdout_only"
    changed.loc[heldout, "string_dtype"] = "unknown_holdout"
    second = AutoML(**options).fit(changed)
    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name
    assert first.best_params == second.best_params
    assert first.feature_schema == second.feature_schema
    assert first.transformed_feature_names == second.transformed_feature_names
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    left = first.best_model.named_steps["preprocessing"].named_transformers_
    right = second.best_model.named_steps["preprocessing"].named_transformers_
    for branch in ["numeric", "boolean", "categorical"]:
        np.testing.assert_array_equal(left[branch].named_steps["imputer"].statistics_,
                                      right[branch].named_steps["imputer"].statistics_)
    for left_categories, right_categories in zip(left["categorical"].named_steps["encoder"].categories_,
                                                right["categorical"].named_steps["encoder"].categories_):
        np.testing.assert_array_equal(left_categories, right_categories)
        assert "holdout_only" not in left_categories and "unknown_holdout" not in left_categories
    if "scaler" in left["numeric"].named_steps:
        np.testing.assert_array_equal(left["numeric"].named_steps["scaler"].mean_,
                                      right["numeric"].named_steps["scaler"].mean_)
        np.testing.assert_array_equal(left["numeric"].named_steps["scaler"].scale_,
                                      right["numeric"].named_steps["scaler"].scale_)
    X = mixed_df.drop(columns="target")
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))


def test_column_empty_in_a_fold_is_an_error_even_when_full_train_has_a_value():
    df, train_rows, _, _ = leakage_dataset()
    df["rare"] = pd.Series([None] * len(df), dtype="string")
    df.loc[train_rows[0], "rare"] = "observed_once"
    assert df.iloc[train_rows].rare.notna().any()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Constant training features")
        with pytest.raises(DataValidationError, match="entirely missing.*rare"):
            AutoML(target="target", models="logistic_regression", preprocessing="basic", cv=3, n_trials=1,
                   random_state=17, verbose=0).fit(df)
