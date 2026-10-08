"""Legacy explicit segmentation, using the corrected binary engine."""

import copy
import pandas as pd
from strategy import local_optimizer
from sco_mod import sco_mod


def second_step(init_model: local_optimizer, col_1: str, col_2: str = None) -> pd.DataFrame:
    """
    Find the best hyperparameters for each model from the chosen scoring with by segmenting the data with two indexes

    Args:
    -----
        init_model (init_model): RationalML object
        col_1 (str): column for first index, first granularity 
        col_2 (str, optional): column for the second index, second granularity. Defaults to None.
    Result:
    -------
    DataFrame: dataframe with metrics for each model

    """
    #Si la colonne 2 non renseigner => pas de tri à deux niveaux
    if col_2 is None:
        col_2 = col_1
    result = pd.DataFrame()
    values_lvl1 = init_model.df[col_1].unique()
    for i in values_lvl1:
        tmp_df_col_1 = init_model.df.copy()
        tmp_df_col_1 = tmp_df_col_1[tmp_df_col_1[col_1] == i]
        values_lvl2 = tmp_df_col_1[col_2].unique()
        for j in values_lvl2:
            tmp_df = tmp_df_col_1.copy()
            tmp_df = tmp_df[tmp_df[col_2] == j]
            tmp_df = tmp_df.drop(list(dict.fromkeys([col_1, col_2])), axis=1)
            tmp_context = copy.copy(init_model)
            tmp_context.df = tmp_df
            tmp_context.num_step = 2
            result = pd.concat([result, sco_mod(tmp_context, i, j)])
    return result.reset_index(drop=True)
