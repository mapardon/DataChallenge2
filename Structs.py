"""
    Small classes (bound to be used in similar way as C struct) whose goal is to give an explicit format of data shared
    among several experiments (clearer than generic lists or tuples).
"""


import os
import statistics
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


"""
    Preprocessing experiments
"""


@dataclass
class PreprocessingParameters:
    numerizer: Literal["remove", "one-hot"] | None = None
    scaler: Literal["minmax"] | None = None
    outlier_detector: Literal["lof"] | None = None
    outlier_detector_nn: int = 0
    remove_uninformative_features: bool = False
    remove_correlated_features: bool = False
    feature_selector: Literal["RFE"] | list | np.ndarray | None = None
    feature_selection_prop: float | None = None

    def __repr__(self):
        out = "Preprocessing params: numerizer: {}, scaler: {}, outlier detector: {} ({}), "
        out += "remove uninformative features: {}, remove correlated features: {}, feature selector: {} ({})"
        return out.format(self.numerizer, self.scaler, self.outlier_detector, self.outlier_detector_nn,
                          self.remove_uninformative_features, self.remove_correlated_features, self.feature_selector,
                          self.feature_selection_prop)


@dataclass
class PreprocessingOutput:
    outliers_detection_res: int | None = None
    uninformative_features: list[str] | None = None
    correlated_features: list[str] | None = None
    selected_features: str | None = None


class PreprocExperimentResult:
    def __init__(self, configuration: PreprocessingParameters, performance: list[float], preprocessing_output: list[PreprocessingOutput]):
        self.configuration: PreprocessingParameters = configuration
        self.f1_scores: list[float] = performance
        self.preprocessing_output: list[PreprocessingOutput] = preprocessing_output

    def __repr__(self):
        out = "\nPreprocessing Experiment\n {}\n performance: {} ({})".format(self.configuration, round(statistics.mean(self.f1_scores), 4), round(statistics.stdev(self.f1_scores), 4))

        if self.configuration.outlier_detector is not None:
            out += "\n n_outliers_removed: {}-{}".format(min(self.preprocessing_output, key=lambda x: x.outliers_detection_res).outliers_detection_res, max(self.preprocessing_output, key=lambda x: x.outliers_detection_res).outliers_detection_res)

        if self.configuration.remove_uninformative_features:
            out += "\n uninformative features removed: {}".format(", ".join(["{} ({})".format(feat, [feat for res in self.preprocessing_output for feat in res.uninformative_features].count(feat)) for feat in set([feat for res in self.preprocessing_output for feat in res.uninformative_features])]))

        if self.configuration.remove_correlated_features:
            out += "\n correlated features removed: {}".format(", ".join(["{} ({})".format(feat, [feat for res in self.preprocessing_output for feat in res.correlated_features].count(feat)) for feat in set([feat for res in self.preprocessing_output for feat in res.correlated_features])]))

        if self.configuration.feature_selector is not None:
            out += "\n features selected: {}".format(", ".join(["{} ({})".format(feat, [feat for res in self.preprocessing_output for feat in res.selected_features].count(feat)) for feat in set([feat for res in self.preprocessing_output for feat in res.selected_features])]))

        return out


"""
    Model experiments
"""

experiment_model_tags = Literal["lm", "dtree", "gbc"]
experiment_model_types = Union[LinearRegression, DecisionTreeClassifier, GradientBoostingClassifier]


@dataclass
class ModelExperimentTagParam:
    model_tag: experiment_model_tags
    model_param: str | None


@dataclass
class ModelExperimentBooter:
    model: ModelExperimentTagParam
    datasets_src: PreprocessingParameters | tuple[os.PathLike, os.PathLike]


@dataclass
class ModelExperimentConfiguration:
    experiment_type: Literal["PI", "SI"] | None = None
    model_tag: experiment_model_tags | None = None
    model: experiment_model_types | None = None
    is_reg_model: bool | None = None
    model_params: dict | None | None = None
    is_bag: bool = False
    estimator_model_tag: experiment_model_tags | None = None

    def __repr__(self):
        out = "{} ({})".format(self.model_tag, self.model_params)
        if self.is_bag:
            out += " @ {}".format(self.estimator_model_tag)
        return out


@dataclass
class ModelExperimentResult:
    config: ModelExperimentConfiguration
    f1: list[float]

    def __repr__(self):
        if len(self.f1) > 1:
            out = "configuration: {}, performance: avg {}, min {}, max {}".format(self.config, round(statistics.mean(self.f1), 4), round(min(self.f1), 4), round(max(self.f1), 4))
        else:
            out = "configuration: {}, performance: {}".format(self.config, round(self.f1[0], 4))
        return out
