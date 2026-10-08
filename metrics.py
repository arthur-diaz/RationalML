"""Legacy metric utilities; the new engine uses evaluation.MetricRegistry."""

import pandas as pd
import numpy as np
from typing import Tuple
from pandas import DataFrame
from pandas.core.series import Series
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
)

from rationalml.exceptions import LegacyAPIError


class Metrics:
    """class of commercial metrics"""

    @staticmethod
    def simple_metrics(y_test: Series, y_pred: np.ndarray):
        """
        calculation of usual metrics
        
        Args:
        -----
            y_test (Series): _description_
            y_pred (np.ndarray): _description_

        Returns:
        --------
            All the usual classification metrics
        """
        accuracy_val = accuracy_score(y_test, y_pred)
        precision_val = precision_score(y_test, y_pred, zero_division=0)
        recall_val = recall_score(y_test, y_pred, zero_division=0)
        fscore_val = f1_score(y_test, y_pred, zero_division=0)

        return accuracy_val, precision_val, recall_val, fscore_val

    @staticmethod
    def feature_importance(model: str, classifier) -> dict:
        """
        Define the feature importance of each model

        Args:
        -----
            model (str): name of the library used 
            classifier (_type_): current model classifier

        Returns:
        --------
            dict: list of feature importance for each model
        """
        if model == "XGBoost":
            feature_imp = classifier.get_booster().get_score(importance_type="total_gain")
            return sorted(feature_imp.items(), key=lambda kv: kv[1], reverse=True)
        if model == "LightGBM":
            name = classifier.booster_.feature_name()
            score = classifier.booster_.feature_importance(importance_type="gain")
            list2, list1 = zip(*sorted(zip(score, name), reverse=True))
            return dict(zip(list1, list2))
        if model == "Sklearn":
            name = classifier.feature_names_in_
            score = classifier.coef_[0]
            list2, list1 = zip(*sorted(zip(score, name), reverse=True))
            coef = dict(zip(list1, list2))
            coef["constant"]=classifier.intercept_[0]
            return coef

    
    @staticmethod
    def apply_threshold(choices: str, y_pred: np.ndarray, y_test, threshold: float, target_name: str
    ) -> Tuple[DataFrame, float]:
        """
        get the classification thresholds for the different types of rates

        Args:
        -----
            choices (str): rate type selected
            y_pred (np.ndarray): predicted values
            y_test (Series): real values
            threshold (float): rate of non holder for the validation
            target_name (str): name of target column

        Returns:
        --------
            Tuple[DataFrame, float]: classification of the prediction with the threshold
        """
        rate_choice = {
            'FP': (False, 0),
            'TP': (False, 1),
            'FN': (True, 1),
            'TN': (True, 0),
            }
        if choices not in rate_choice or not 0 <= threshold <= 1:
            raise ValueError("A valid rate choice and a threshold rate in [0, 1] are required.")
        threshold_type = rate_choice[choices]
        y_pred = np.asarray(y_pred)
        y_test = np.asarray(y_test)
        if y_pred.ndim != 1 or y_test.ndim != 1 or len(y_pred) != len(y_test):
            raise ValueError("Predictions and targets must be aligned one-dimensional arrays.")
        index = int(np.count_nonzero(y_test == threshold_type[1]))
        if index == 0:
            raise ValueError("The selected rate requires examples of its target class.")
        df_threshold = (
            pd.DataFrame({0: y_pred, 1: y_test})
            .sort_values(by=[0], ascending=threshold_type[0])
            .reset_index(drop=True)
        )
        df_threshold = df_threshold[df_threshold[1] == threshold_type[1]]
        threshold = min(int(np.ceil(index * threshold)), index - 1)
        df_threshold = df_threshold[0].iloc[threshold]
        y_pred_test = (y_pred > df_threshold).astype(int)
        return y_pred_test, df_threshold

    @staticmethod
    def sub_area(
        threshold: float, y_test: Series, y_pred: np.ndarray, nb_bins: int = 800
                ) -> float:
        """Retired: the historical partial ROC area has no reliable contract."""
        raise LegacyAPIError(
            "Metrics.sub_area is retired in V0.1.1: it discarded probabilities before "
            "constructing the ROC curve, and the partial-area boundary/normalization "
            "contract is undefined. Use the registered roc_auc metric for full ROC AUC."
        )

    @staticmethod
    def sub_area_pr(
        threshold_recall: float, y_test: Series, y_pred: np.ndarray
                    ) -> float:
        """Retired: partial PR integration and its boundary are underspecified."""
        raise LegacyAPIError(
            "Metrics.sub_area_pr is retired in V0.1.1: the historical integration "
            "included intervals beyond the recall boundary, and the interpolation/"
            "normalization contract is undefined. The registered average_precision "
            "metric is available for full-curve evaluation, not as an equivalent partial area."
        )
