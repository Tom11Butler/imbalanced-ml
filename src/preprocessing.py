"""
Author: TOM BUTLER

Helper file for useful preprocessing functions.

"""

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector


def categorical_transformer():
    """
    Preprocessing pipeline for categorical data
    """
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]

    )

    return cat_transformer


def numeric_transformer(scaling=False):
    """
    Preprocessing pipeline for numeric data
    """

    if scaling:
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", RobustScaler())
            ]
        )
    else:
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
            ]
        )

    return num_transformer


def column_transformer(scaling=False):
    """
    Transforms both numeric and categorical features
    """

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer(scaling),
             make_column_selector(dtype_include=np.number)),
            ("cat", categorical_transformer(),
             make_column_selector(dtype_include=object))
        ],
        n_jobs=2
    )

    return column_transformer


def preprocessor_pipeline(scaling=False):
    return Pipeline(steps=[("columntransformer", column_transformer(scaling))])

def rebalance_dataset(X, y, ratio=1):
    """
    Re-balances the class ratio in the dataset.

    :param X: pandas DataFrame of the feature data
    :param y: the labels
    :param ratio: float ratio of positive:negative cases
    :return: pandas DataFrame of rebalanced data
    """

    # TODO: count number of positive and negative samples

    # TODO: see if it is possible to use all positive in re-balance

    # TODO: take appropriate number of positive/negative samples for ratio

    # TODO: send back a new X, y to train on

    return X, y
