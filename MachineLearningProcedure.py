from typing import Literal

import pandas as pd

from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing, PreprocessingParameters
from ParametricIdentification import ParametricIdentification

FEATURES_SRC = "data/train_features_short.csv"
LABELS_SRC = "data/train_labels_short.csv"
CHALLENGE_SRC = None


class MachineLearningProcedure:
    def __init__(self, preproc_pars: PreprocessingParameters | None, model_tags: list[Literal["lm", "tree"]] | None):
        self.preproc_pars: PreprocessingParameters | None = preproc_pars
        self.model_tags: list[Literal["lm", "tree"]] | None = model_tags

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
        if self.train_features is None:  # precomputed preprocessed datasets
            self.train_features, self.train_labels, self.validation_features, self.validation_labels = None, None, None, None
            print("/!/ Datasets not loaded")
            return

        pi = ParametricIdentification(self.train_features, self.train_labels, self.validation_features, self.validation_labels, 5)
        for res in pi.parametric_identification(self.model_tags):
            print(res)

    def model_exploitation(self):
        pass
