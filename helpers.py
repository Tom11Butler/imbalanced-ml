import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector


class CustomImputer(SimpleImputer):
    """
    Extends the imputer class to include feature names. Addresses original
    imputer bug not filling nans
    """

    def fit(self, X, y=None):
        self.cols = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return np.where(X.isna(), self.fill_value, X)

    def get_feature_names(self):
        return self.cols


# preprocessing the training data to make an actually good model

# preprocessing the training data to make an actually good model

def categorical_transformer():
    """
    Preprocessing pipeline for categorical data
    """
    cat_transformer = Pipeline(
        steps=[
            ("imputer", CustomImputer(strategy="constant", fill_value="missing")),
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
                ("imputer", CustomImputer(strategy="constant", fill_value=0)),
                ("scaler", RobustScaler())
            ]
        )
    else:
        num_transformer = Pipeline(
            steps=[
                ("imputer", CustomImputer(strategy="constant", fill_value=0)),
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