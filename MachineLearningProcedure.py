import statistics
from math import ceil
from typing import Literal

import numpy as np
import pandas as pd

from DataAnalysis import DataAnalysis
from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing, PreprocessingParameters
from ParametricIdentificationCV import ModelExperimentResult
from ParametricIdentification import ParametricIdentification
from PreprocessingIdentification import PreprocessingIdentification, PreprocExperimentResult
from StructuralIdentification import StructuralIdentification

SHORT = True

if SHORT:
    FEATURES_SRC = "data/train_features_short.csv"
    LABELS_SRC = "data/train_labels_short.csv"
    CHALLENGE_SRC = "data/train_features_short.csv"

else:
    FEATURES_SRC = "data/train_features.csv"
    LABELS_SRC = "data/train_labels.csv"
    CHALLENGE_SRC = "data/challenge_features.csv"


class MachineLearningProcedure:
    def __init__(self, preproc_params: PreprocessingParameters | None, mi_models: list[Literal["lm", "dtree"]] | None):
        self.preproc_params: PreprocessingParameters | None = preproc_params  # can be specified to skip preproc identification
        self.mi_models: list[Literal["lm", "dtree"]] | None = mi_models
        self.pi_candidates: list[ModelExperimentResult] = list()
        self.final_model_candidate: ModelExperimentResult | None = None

        self.train_features: pd.DataFrame | None = None
        self.train_labels: pd.DataFrame | None = None
        self.validation_features: pd.DataFrame | None = None
        self.validation_labels: pd.DataFrame | None = None

    def main(self, modes: list[Literal["DA", "PPI", "PI", "SI", "ME"]]):
        if "DA" in modes:
            DataAnalysis().exploratory_analysis()

        if "PPI" in modes:
            self.preprocessing_identification()

        if "PI" in modes:
            if self.mi_models is None:
                raise Warning("No model has been provided for model identification")
            self.parametric_identification()

        if "SI" in modes:
            self.structural_identification()

        if "ME" in modes:
            self.model_exploitation()

    # PREPROCESSING IDENTIFICATION

    def preprocessing_identification(self):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        features, labels = dl.get_train_dataset()

        ppi = PreprocessingIdentification(features, labels, cv_folds=3)
        ppi_candidates: list[PreprocExperimentResult] = list()
        tmp: list[PreprocExperimentResult] = list()
        best_preproc_params_combination: PreprocessingParameters = PreprocessingParameters()

        default_numerizer: Literal["remove", "one-hot"] = "remove"
        default_scaler = None
        default_outlier_detector = None
        default_remove_uninf_features = False
        default_remove_corr_features = False
        default_feature_selector = None
        default_feat_select_prop = None

        # Annoying thing to satisfy type checking
        numerizers: list[Literal["remove", "one-hot"]] = ["remove", "one-hot"]
        scalers: list[Literal["minmax"]] = ["minmax"]
        feature_selectors: list[Literal["RFE"]] = ["RFE"]
        feat_select_props: list[float] = [1/4, 1/2, 3/4]

        # Categorical variables numerizer
        for numerizer in numerizers:
            ppars = PreprocessingParameters(numerizer, default_scaler, default_outlier_detector, default_remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.numerizer = max(tmp, key=lambda x: statistics.mean(x.f1_scores)).configuration.numerizer
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Numerical features scaling
        for scaler in scalers:
            ppars = PreprocessingParameters(default_numerizer, scaler, default_outlier_detector, default_remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.scaler = max(tmp, key=lambda x: statistics.mean(x.f1_scores)).configuration.scaler
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Remove uninformative features
        for remove_uninf_features in [True]:
            ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.remove_uninformative_features = max(tmp, key=lambda x: statistics.mean(x.f1_scores)).configuration.remove_uninformative_features
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Remove correlated features
        for remove_corr_features in [True]:
            ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_remove_uninf_features, remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.remove_correlated_features = max(tmp, key=lambda x: statistics.mean(x.f1_scores)).configuration.remove_correlated_features
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Feature selection
        for feature_selector in feature_selectors:
            if feature_selector in ["RFE"]:
                for feat_select_prop in feat_select_props:
                    ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_remove_uninf_features, default_remove_corr_features, feature_selector, feat_select_prop)
                    tmp.append(ppi.preprocessing_identification(ppars))
            else:
                ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_remove_uninf_features, default_remove_corr_features, feature_selector, default_feat_select_prop)
                tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.feature_selector, best_preproc_params_combination.feature_selection_prop = max(tmp, key=lambda x: statistics.mean(x.f1_scores)).configuration.feature_selector, max(tmp, key=lambda x: statistics.mean(x.f1_scores)).configuration.feature_selection_prop
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Combination of better-performing features
        ppi_candidates.append(ppi.preprocessing_identification(best_preproc_params_combination))

        # Keep config having shown best results & display results
        ppi_candidates.sort(reverse=True, key=lambda x: statistics.mean(x.f1_scores))
        self.preproc_params = ppi_candidates[0].configuration

        print("\n * Preprocessing Identification *")
        for ppi_c in ppi_candidates:
            print(ppi_c)

    # MODEL IDENTIFICATION

    def load_and_preprocess_dataset(self, config: PreprocessingParameters):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        train_features, train_labels = dl.get_train_dataset()

        dp = DataPreprocessing(train_features, train_labels, None)
        dp.preprocessing(config)
        self.train_features, self.train_labels, self.validation_features, self.validation_labels = dp.get_train_validation_datasets()

    def parametric_identification(self):
        # TODO several experiments with reshuffling (?)

        # Load data
        if self.preproc_params is not None:
            self.load_and_preprocess_dataset(self.preproc_params)

        else:  # load precomputed preprocessed datasets
            self.train_features, self.train_labels, self.validation_features, self.validation_labels = None, None, None, None
            print("/!/ Datasets not loaded")
            return

        # Parametric identification
        pi = ParametricIdentification(self.train_features, self.train_labels, 5)
        self.pi_candidates = pi.parametric_identification(self.mi_models)

        print("\n * Parametric Identification *")
        for pi_c in self.pi_candidates:
            print(pi_c)

    def structural_identification(self):
        """ Tournament between the best MI candidates on unused validation set """

        # Load data (incase data loaded during parametric identification)
        if self.train_features is None:
            if self.preproc_params is not None:
                self.load_and_preprocess_dataset(self.preproc_params)

            else:  # load precomputed preprocessed datasets
                self.train_features, self.train_labels, self.validation_features, self.validation_labels = None, None, None, None
                print("/!/ Datasets not loaded")
                return

        # Structural identification
        si = StructuralIdentification(self.train_features, self.train_labels, self.validation_features, self.validation_labels)
        si_candidates = si.model_selection(self.pi_candidates[:ceil(len(self.pi_candidates) * 0.25)])
        self.final_model_candidate = si_candidates[0]

        print("\n * Structural Identification *")
        for si_c in si_candidates:
            print(si_c)

    # MODEL EXPLOITATION

    def model_exploitation(self):
        """ Use the models and preprocessing parameters having shown the best performance during training
        to predict challenge data """

        # load challenge data
        dl = DataLoading()
        dl.load_data(CHALLENGE_SRC, None, False)
        features, data_id = dl.get_challenge_dataset()

        # preprocess challenge data
        dp = DataPreprocessing(features, None, data_id)
        dp.preprocessing(self.preproc_params)
        features, data_id = dp.get_challenge_dataset()

        # predict challenge data
        model, is_reg_model = self.final_model_candidate.model, self.final_model_candidate.is_reg_model
        # what are you doing linear model ??
        labels = np.clip(np.round(model.predict(features)).astype(int), 1, 3).flatten() if is_reg_model else model.predict(features).astype(int)

        # store challenge data
        pd.DataFrame({
            "building_id": data_id,
            "damage_grade": labels
        }).to_csv("data/submission.csv", index=False)

        print("\n * Model exploitation complete")
