"""
Author: TOM BUTLER

Helper file for useful model functions.

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)


def create_model(params):
    """
    Creates and returns a model object with set parameters.

    :param params: parameter dictionary for the model
    :return: model object
    """

    model = RandomForestClassifier(*params)

    return model


def train_model(model, X, y):
    """

    :param model:
    :param X:
    :param y:
    :return: model
    """
    model.fit(X, y)
    return model


def score_model(model, test):
    """

    :param model: model object to score test data on
    :param test: test data, both features and labels
    :return: pandas DataFrame of metrics
    """

    # TODO: score the test data with the model
    scores = model.predict(test)

    # TODO: get metrics to add to the report

    # TODO: return the metric dataframe and the scores

    df_metrics = pd.DataFrame()

    return df_metrics, scores
