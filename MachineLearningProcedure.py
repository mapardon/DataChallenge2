from typing import Literal

import pandas as pd
from sklearn.linear_model import LinearRegression

from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing, PreprocessingParameters
from ParametricIdentification import ParametricIdentification

FEATURES_SRC = "data/train_features_short.csv"
LABELS_SRC = "data/train_labels_short.csv"
CHALLENGE_SRC = None


class MachineLearningProcedure:
    def __init__(self, preproc_pars: PreprocessingParameters):
        self.preproc_pars: PreprocessingParameters = preproc_pars

        self.train_features: pd.DataFrame | None = None
        self.train_labels: pd.DataFrame | None = None
        self.validation_features: pd.DataFrame | None = None
        self.validation_labels: pd.DataFrame | None = None

    def main(self, modes: list[Literal["PI", "MI", "ME"]]):
        if "PI" in modes:
            self.preprocessing_identification()

        if "MI" in modes:
            self.model_identification()

        if "ME" in modes:
            self.model_exploitation()

    def preprocessing_identification(self):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        train_features, train_labels = dl.get_train_dataset()

        dp = DataPreprocessing(train_features, train_labels, None)
        dp.preprocessing(self.preproc_pars)
        self.train_features, self.train_labels, self.validation_features, self.validation_labels = dp.get_train_validation_datasets()

    def model_identification(self):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        train_features, train_labels = dl.get_train_dataset()
        train_features, train_labels, validation_features, validation_labels = dp.get_train_validation_datasets()

        pi = ParametricIdentification(train_features, train_labels, validation_features, validation_labels, 5)
        m = LinearRegression()
        print(pi.parametric_identification(m, True))

    def model_exploitation(self):
        pass


if __name__ == '__main__':

    MachineLearningProcedure()
