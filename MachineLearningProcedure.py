import copy
import os
import pathlib
import statistics
import sys
from math import ceil
from multiprocessing import Process, Queue
from typing import Literal

import numpy as np
import pandas as pd

from DataAnalysis import DataAnalysis
from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing
from ParametricIdentification import ParametricIdentification
from PreprocessingIdentification import PreprocessingIdentification
from StructuralIdentification import StructuralIdentification
from Structs import ModelExperimentBooter, PreprocessingParameters, ModelExperimentResult, PreprocExperimentResult, \
    ModelExperimentTagParam, experiment_result_sorting_param

SHORT = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True

if SHORT:
    FEATURES_SRC: os.PathLike = pathlib.Path("data/train_features_short.csv")
    LABELS_SRC: os.PathLike = pathlib.Path("data/train_labels_short.csv")
    CHALLENGE_SRC: os.PathLike = pathlib.Path("data/train_features_short.csv")

else:
    FEATURES_SRC: os.PathLike = pathlib.Path("data/train_features.csv")
    LABELS_SRC: os.PathLike = pathlib.Path("data/train_labels.csv")
    CHALLENGE_SRC: os.PathLike = pathlib.Path("data/challenge_features.csv")


class MachineLearningProcedure:
    def __init__(self, preproc_params: PreprocessingParameters | None, mi_configs: ModelExperimentBooter | None):
        self.preproc_params: PreprocessingParameters | None = preproc_params  # can be specified to skip preproc identification
        self.mi_configs: ModelExperimentBooter | None = mi_configs
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
            if self.mi_configs is None:
                raise Warning("No model has been provided for model identification")

            # Load data
            self.model_experiment_data_prep(self.mi_configs.preproc_params, self.mi_configs.datasets_src)

            # Run experiments
            procs: list[Process] = list()
            pipe: Queue = Queue()
            for mi_config in self.mi_configs.model_tag_param:
                procs.append(Process(target=self.parametric_identification_wrapper, args=(mi_config, pipe)))

            [p.start() for p in procs]
            print("1")
            [p.join() for p in procs]

            print("2")

            while not pipe.empty():
                self.pi_candidates.extend(pipe.get())

        if "SI" in modes:
            self.structural_identification()

        if "ME" in modes:
            self.model_exploitation()

    # PREPROCESSING IDENTIFICATION

    def preprocessing_identification(self) -> None:
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        features, labels = dl.get_train_dataset()

        ppi = PreprocessingIdentification(features, labels, cv_folds=3)
        ppi_candidates: list[PreprocExperimentResult] = list()
        tmp: list[PreprocExperimentResult] = list()
        best_preproc_params_combination: PreprocessingParameters = PreprocessingParameters()

        default_numerizer: Literal["remove", "one-hot"] = "one-hot"
        default_scaler = None
        default_outlier_detector = None
        default_outlier_detector_nn = int()
        default_remove_uninf_features = False
        default_remove_corr_features = False
        default_feature_selector = None
        default_feat_select_prop = None

        # Annoying thing to satisfy type checking
        numerizers: list[Literal["remove", "one-hot"]] = ["remove", "one-hot"]
        scalers: list[Literal["minmax"]] = ["minmax"]
        outlier_detectors: list[Literal["lof"]] = ["lof"]
        feature_selectors: list[Literal["RFE"]] = ["RFE"]
        feat_select_props: list[float] = [1/4, 1/2, 3/4]

        get_best_config = lambda l: max(l, key=lambda x: statistics.mean(x.f1_scores)).configuration

        # Categorical variables numerizer
        for numerizer in numerizers:
            ppars = PreprocessingParameters(numerizer, default_scaler, default_outlier_detector, default_outlier_detector_nn, default_remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.numerizer = get_best_config(tmp).numerizer
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Numerical features scaling
        for scaler in scalers:
            ppars = PreprocessingParameters(default_numerizer, scaler, default_outlier_detector, default_outlier_detector_nn, default_remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.scaler = get_best_config(tmp).scaler
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Outlier detectors
        for outlier_detector in outlier_detectors:
            for nn in [2, 3, 5, 10, 25]:
                ppars = PreprocessingParameters(default_numerizer, default_scaler, outlier_detector, nn, default_remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
                tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.outlier_detector, best_preproc_params_combination.outlier_detector_nn = get_best_config(tmp).outlier_detector, get_best_config(tmp).outlier_detector_nn
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Remove uninformative features
        for remove_uninf_features in [True]:
            ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_outlier_detector_nn, remove_uninf_features, default_remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.remove_uninformative_features = get_best_config(tmp).remove_uninformative_features
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Remove correlated features
        for remove_corr_features in [True]:
            ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_outlier_detector_nn, default_remove_uninf_features, remove_corr_features, default_feature_selector, default_feat_select_prop)
            tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.remove_correlated_features = get_best_config(tmp).remove_correlated_features
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Feature selection
        for feature_selector in feature_selectors:
            if feature_selector in ["RFE"]:
                for feat_select_prop in feat_select_props:
                    ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_outlier_detector_nn, default_remove_uninf_features, default_remove_corr_features, feature_selector, feat_select_prop)
                    tmp.append(ppi.preprocessing_identification(ppars))
            else:
                ppars = PreprocessingParameters(default_numerizer, default_scaler, default_outlier_detector, default_outlier_detector_nn, default_remove_uninf_features, default_remove_corr_features, feature_selector, default_feat_select_prop)
                tmp.append(ppi.preprocessing_identification(ppars))
        best_preproc_params_combination.feature_selector, best_preproc_params_combination.feature_selection_prop = get_best_config(tmp).feature_selector, get_best_config(tmp).feature_selection_prop
        ppi_candidates.extend(tmp)
        tmp.clear()

        # Combination of best-performing features
        ppi_candidates.append(ppi.preprocessing_identification(best_preproc_params_combination))

        # Store config having shown best results
        ppi_candidates.sort(**experiment_result_sorting_param)
        self.preproc_params = ppi_candidates[0].configuration
        self.preproc_params.feature_selector = ppi_candidates[0].preprocessing_output[0].selected_features

        # Display results
        print("\n * Preprocessing Identification *")
        for ppi_c in ppi_candidates:
            print(ppi_c)

    # MODEL IDENTIFICATION

    def load_datasets(self, features_src: os.PathLike | None = None, labels_src: os.PathLike | None = None) -> None:
        """ NB: If this method (and so does load_and_preprocessing_datasets) is called from parametric identification,
        datasets will be split between train and validations sets, even if validation sets will not be used in the
        cross-validation loop. This is done to perform parametric identification and structural identification in the
        same conditions (i.e., models trained more in parametric identification). """

        dl = DataLoading()
        if None not in [features_src, labels_src]:
            dl.load_data(features_src, labels_src, True)
        else:
            dl.load_data(FEATURES_SRC, LABELS_SRC, True)
        self.train_features, self.train_labels, self.validation_features, self.validation_labels = DataPreprocessing(*dl.get_train_dataset(), None).get_train_validation_datasets()

    def load_and_preprocess_datasets(self, config: PreprocessingParameters) -> None:
        self.load_datasets()

        dp = DataPreprocessing(self.train_features, self.train_labels, None)
        dp.preprocessing(config)
        self.train_features, self.train_labels, self.validation_features, self.validation_labels = dp.get_train_validation_datasets()

    def model_experiment_data_prep(self, preproc_params: PreprocessingParameters = None, datasets_src: tuple[os.PathLike, os.PathLike] = None) -> None:
        """ Prepare datasets for parametric [and structural] identification. This is performed in an independent method
         to prepare datasets before the actual parametric identification step and therefore avoid recomputing them for
         each experiment given they're computed in independent sub-processes. """

        # TODO several experiments with reshuffling (?)

        # Load data
        if any(_ is None for _ in [self.train_features, self.train_labels]):

            if self.preproc_params is not None:  # use result of preprocessing identification or user specified preprocessing parameters
                self.load_and_preprocess_datasets(self.preproc_params)

            elif preproc_params is not None:
                self.load_and_preprocess_datasets(preproc_params)

            elif datasets_src is not None:  # load precomputed preprocessed datasets
                self.load_datasets(*datasets_src)

            else:
                raise Warning("Couldn't load datasets for parametric identification")

    def parametric_identification_wrapper(self, model_tag_param: ModelExperimentTagParam, pipe: Queue):
        self.parametric_identification(model_tag_param)
        pipe.put(copy.deepcopy(self.pi_candidates))

    def parametric_identification(self, model_tag_param: ModelExperimentTagParam):

        # Parametric identification
        pi = ParametricIdentification(self.train_features, self.train_labels, 5)
        self.pi_candidates = pi.parametric_identification(model_tag_param)

        print("\n * Parametric Identification *")
        for pi_c in sorted(self.pi_candidates, **experiment_result_sorting_param):
            print(pi_c)

    def structural_identification(self) -> None:
        """ Tournament between the best MI candidates on unused validation set. Should be executed after running
         parametric_identification. """

        # Load data (incase data not loaded during parametric identification)
        if any(_ is None for _ in [self.train_features, self.train_labels, self.validation_features, self.validation_labels]):

            if self.preproc_params is not None:
                self.load_and_preprocess_datasets(self.preproc_params)

            else:
                raise Warning("Data couldn't be loaded before structural identification.")

        # Structural identification
        si = StructuralIdentification(self.train_features, self.train_labels, self.validation_features, self.validation_labels)
        self.pi_candidates.sort(**experiment_result_sorting_param)
        si_candidates = si.model_selection([pi_candidate.config for pi_candidate in self.pi_candidates[:ceil(len(self.pi_candidates) * 0.25)]])
        si_candidates.sort(**experiment_result_sorting_param)
        self.final_model_candidate = si_candidates[0]

        print("\n * Structural Identification *")
        for si_c in si_candidates:
            print(si_c)

    # MODEL EXPLOITATION

    def model_exploitation(self) -> None:
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
        model, is_reg_model = self.final_model_candidate.config.model, self.final_model_candidate.config.is_reg_model
        labels = np.clip(np.round(model.predict(features)).astype(int), 1, 3).flatten() if is_reg_model else model.predict(features).astype(int)

        # store challenge data
        pd.DataFrame({
            "building_id": data_id,
            "damage_grade": labels
        }).to_csv("data/submission.csv", index=False)

        print("\n * Model exploitation complete")
