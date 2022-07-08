"""
Author: TOM BUTLER

Helper file for useful preprocessing functions.

"""

import numpy as np
from collections import Counter

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


def rebalance_dataset(df, label, ratio=1):
    """
    Re-balances the class ratio in the dataset.

    :param df: pandas DataFrame of the data
    :param label: string of the label column identifier
    :param ratio: float ratio of positive:negative cases
    :return: pandas DataFrame of rebalanced data
    """

    label_counts = Counter(df[label])

    # print("Rebalancing the data ...")

    if label_counts[1] * ratio <= label_counts[0]:
        # keep all positive instances, drop negative instances
        pos_sample_indices = df.query(f"{label} == 1").index.values

        n_neg_samples = round(label_counts[1] * ratio)
        # print("Number of negative samples required: {:,}".format(n_neg_samples))

        # get the number of negative samples needed
        neg_sample_indices = df.query(f"{label} == 0").sample(n=n_neg_samples).index.values

    else:
        # keep all negative instances, start dropping positive instances
        neg_sample_indices = df.query(f"{label} == 0").index.values

        n_pos_samples = round(label_counts[0] / ratio)
        # print("Number of positive samples required: {:,}".format(n_pos_samples))

        # get the number of negative samples needed
        pos_sample_indices = df.query(f"{label} == 1").sample(n=n_pos_samples).index.values

    sample_indices = np.concatenate((pos_sample_indices, neg_sample_indices))

    df_sample = df.query("index in @sample_indices").copy()
    # label_counts = Counter(df_sample[label])
    # print(label_counts)

    return df_sample

