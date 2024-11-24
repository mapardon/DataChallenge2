import pandas as pd

from ModelIdentification import ModelIdentification


class ParametricIdentification(ModelIdentification):
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, test_features, test_labels, cv_folds)
