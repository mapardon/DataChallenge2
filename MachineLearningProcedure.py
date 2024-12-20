from typing import Literal

import pandas as pd

from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing, PreprocessingParameters
from ParametricIdentificationCV import ModelExperimentResult
from ParametricIdentification import ParametricIdentification
from PreprocessingIdentification import PreprocessingIdentification
from StructuralIdentification import StructuralIdentification

FEATURES_SRC = "data/train_features_short.csv"
LABELS_SRC = "data/train_labels_short.csv"
CHALLENGE_SRC = None


class MachineLearningProcedure:
    def __init__(self, pi_confs: list[PreprocessingParameters] | None, mi_models: list[Literal["lm", "dtree"]] | None,
                 preproc_pars: PreprocessingParameters | None):
        self.pi_confs: list[PreprocessingParameters] | None = pi_confs
        self.mi_models: list[Literal["lm", "dtree"]] | None = mi_models
        self.preproc_pars: PreprocessingParameters | None = preproc_pars

        self.train_features: pd.DataFrame | None = None
        self.train_labels: pd.DataFrame | None = None
        self.validation_features: pd.DataFrame | None = None
        self.validation_labels: pd.DataFrame | None = None
        self.final_candidate: ModelExperimentResult | None = None

    def main(self, modes: list[Literal["PPI", "MI", "ME"]]):
        if "PPI" in modes:
            if self.pi_confs is None:
                raise Warning("No configuration has been provided for preprocessing identification")
            self.preprocessing_identification()

        if "MI" in modes:
            if self.mi_models is None:
                raise Warning("No model has been provided for model identification")
            self.model_identification()

        if "ME" in modes:
            self.model_exploitation()

    def load_and_preprocess_dataset(self):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        train_features, train_labels = dl.get_train_dataset()

        dp = DataPreprocessing(train_features, train_labels, None)
        dp.preprocessing(self.preproc_pars)
        self.train_features, self.train_labels, self.validation_features, self.validation_labels = dp.get_train_validation_datasets()

    def preprocessing_identification(self):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        features, labels = dl.get_train_dataset()

        ppi_candidates = PreprocessingIdentification(self.pi_confs, features, labels, cv_folds=3).preprocessing_identification()
        print("\n * Preprocessing Identification *")
        for ppi_c in ppi_candidates:
            print(ppi_c)

    def model_identification(self):
        # Pre-process data
        if self.preproc_pars is not None:
            self.load_and_preprocess_dataset()

        else:  # load precomputed preprocessed datasets
            self.train_features, self.train_labels, self.validation_features, self.validation_labels = None, None, None, None
            print("/!/ Datasets not loaded")
            return

        # Model identification
        pi = ParametricIdentification(self.train_features, self.train_labels, 5)
        pi_candidates: list[ModelExperimentResult] = pi.parametric_identification(self.mi_models)
        print("\n * Parametric Identification *")
        for pi_c in pi_candidates:
            print(pi_c)

        # Structural identification = tournament between the best MI candidates on unused validation set
        si = StructuralIdentification(self.train_features, self.train_labels, self.validation_features, self.validation_labels, 5)
        si_candidates = si.model_selection(pi_candidates)
        self.final_candidate = si_candidates[0]
        print("\n * Structural Identification *")
        for si_c in si_candidates:
            print(si_c)

    def model_exploitation(self):
        pass
