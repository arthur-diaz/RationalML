"""Legacy configuration holder; use rationalml.AutoML for new code."""

import warnings
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from rationalml.optimization.spaces import legacy_search_domains


class local_optimizer:
    def __init__(
        self,
        df: DataFrame,
        target_name: str,
        nb_cv: int = 3,
        seuil_test: float = 0.70,
        seuil_val: float = 0.025,
        score: list = None,
        algo="all",
        nb_iter: int = 30,
        test_size: float = 0.3,
        nunique_min: int = 25,
        nbins: int = 10,
        class_weight: int = 1,
        num_step: int = 1,
        changes: list = None,
        rate_type: str = "FP",
        threshold_type: str = "val",
        excel_report: bool = False,
        pruner : bool = True,
        verbose: int = 1,
        seed: int = 7,
        timeout: int = None,
        n_jobs: int = 1,
        n_estimators: int = 100,
        early_stopping_rounds: int = 10,
        multi_regression: bool = True,
        model_save: bool = None,
        model_select: str = None,
    ):
        warnings.warn("local_optimizer is legacy; migrate to AutoMLConfig and AutoML.",
                      DeprecationWarning, stacklevel=2)
        self.df = df.copy(deep=True)
        self.target_name = target_name
        self.nb_cv = nb_cv
        self.seuil_test = seuil_test
        self.seuil_val = seuil_val
        self.score = list(score) if score is not None else ["precision"]
        
        if algo == "all":
            self.algo = ["xgb_gb", "lgb_gb", "xgb_rf", "lgb_rf", "xgb_mx", "skl_rl", "skl_rs"]
        elif type(algo)== str:
            self.algo = algo.split()
        else:
            self.algo = algo
            
        self.nb_iter = nb_iter
        self.test_size = test_size
        self.nunique_min = nunique_min
        self.nbins = nbins
        self.class_weight = class_weight
        self.num_step = num_step
        self.rate_type = rate_type
        self.threshold_type = threshold_type
        self.excel_report = excel_report
        self.pruner = pruner
        self.verbose = verbose
        self.seed = seed
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.multi_regression = multi_regression
        self.model_save = model_save
        self.model_select = model_select
        
        #search domain for each algorithm
        params = legacy_search_domains()
    
        
        #apply user-defined domain changes
        if changes is not None:
            for i in changes:
                if i[0] not in params:
                    raise ValueError(f"Model {i[0]!r} not found")
                else:
                    if i[1] not in params[i[0]]:
                        raise ValueError(f"Parameter {i[1]!r} not found")
                    else: params[i[0]][i[1]]=(i[2], i[3])
        self.h_param = params
            
    def _pandas_sets(self):
        df = self.df.copy(deep=True)
        X = df.drop(columns=[self.target_name])
        Y = df[self.target_name].astype("category").cat.codes
        return X, Y

    def discretize(self, df: DataFrame) -> DataFrame:
        """
        discretization of continuous variables

        Args:
        -----
                df (DataFrame): Dataframe to discretize
                nbins (int, optional): Quantile number for categorization. Defaults to 10.
                nunique_min (int, optional): Minimum number of different values to consider a continuous variable. Defaults to 20.

        Returns:
        --------
                DataFrame: descretize DataFrame
        """
        df = df.copy(deep=True)
        if self.verbose > 0:
            iteration = tqdm(df.columns)
        else:
            iteration = df.columns
        for i in iteration:
            if i != "id":
                if df[i].nunique() >= self.nunique_min:
                    df[i] = pd.qcut(
                        df[i], self.nbins, precision=10, labels=False, duplicates="drop"
                    )
        return df
