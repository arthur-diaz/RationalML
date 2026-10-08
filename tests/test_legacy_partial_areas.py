import numpy as np
import pandas as pd
import pytest

from metrics import Metrics
from rationalml.exceptions import LegacyAPIError


@pytest.mark.parametrize("method_name", ["sub_area", "sub_area_pr"])
@pytest.mark.parametrize("labels, probabilities", [
    ([0, 0, 1, 1], [0.1, 0.4, 0.7, 0.9]),
    ([1, 1], [0.2, 0.8]),
    ([], []),
])
def test_legacy_partial_areas_are_explicitly_retired(method_name, labels, probabilities):
    y, proba = pd.Series(labels), np.array(probabilities)
    with pytest.raises(LegacyAPIError, match=rf"Metrics\.{method_name} is retired in V0\.1\.1"):
        getattr(Metrics, method_name)(0.5, y, proba)


def test_partial_roc_signature_still_accepts_historical_keyword_arguments():
    with pytest.raises(LegacyAPIError, match="partial-area.*undefined"):
        Metrics.sub_area(threshold=0.5, y_test=pd.Series([0, 1]), y_pred=np.array([0.1, 0.9]), nb_bins=10)


def test_partial_pr_signature_still_accepts_historical_keyword_arguments():
    with pytest.raises(LegacyAPIError, match="interpolation/normalization.*undefined"):
        Metrics.sub_area_pr(threshold_recall=0.8, y_test=pd.Series([0, 1]), y_pred=np.array([0.1, 0.9]))
