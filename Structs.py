"""
    Small classes (bound to be used in similar way as C struct) whose goal is to give an explicit format of data shared
    among several experiments (clearer than generic lists or tuples).
"""


import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Literal, Union

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

"""
    Paths
"""

SHORT = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
PREPROC = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True

if SHORT:
    FEATURES_SRC: pathlib.Path = pathlib.Path("data/train_features_short.csv")
    LABELS_SRC: pathlib.Path = pathlib.Path("data/train_labels_short.csv")
    CHALLENGE_SRC: pathlib.Path = pathlib.Path("data/train_features_short.csv")
else:
    FEATURES_SRC: pathlib.Path = pathlib.Path("data/train_features.csv")
    LABELS_SRC: pathlib.Path = pathlib.Path("data/train_labels.csv")
    CHALLENGE_SRC: pathlib.Path = pathlib.Path("data/challenge_features.csv")

if PREPROC:
    if SHORT:
        FEATURES_SRC = pathlib.Path("data/train_features_preprocessed_short.csv")
        LABELS_SRC = pathlib.Path("data/train_labels_preprocessed_short.csv")
        CHALLENGE_SRC: pathlib.Path = pathlib.Path("data/challenge_features_preprocessed_short.csv")
    else:
        FEATURES_SRC = pathlib.Path("data/train_features_preprocessed.csv")
        LABELS_SRC = pathlib.Path("data/train_labels_preprocessed.csv")
        CHALLENGE_SRC: pathlib.Path = pathlib.Path("data/challenge_features_preprocessed.csv")

STORAGE = "save-models"


"""
    Preprocessing experiments
"""

experiment_result_sorting_param = {"reverse": True, "key": lambda x: statistics.mean(x.f1_scores)}


@dataclass
class PreprocessingExperimentBooter:
    n_exp: int
    features_path: pathlib.Path
    labels_path: pathlib.Path


@dataclass
class PreprocessingParameters:
    numerizer: Literal["remove", "one-hot"] | None = None
    scaler: Literal["minmax"] | None = None
    outlier_detector: Literal["lof"] | None = None
    outlier_detector_nn: int = 0
    remove_uninformative_features: bool = False
    remove_correlated_features: bool = False
    feature_selector: Literal["RFE"] | list | None = None
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
    selected_features: list[str] | None = None


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

experiment_model_tags = Literal["lm", "dtree", "rforest", "gbc", "knn", "svm"]
experiment_model_types = Union[LinearRegression, DecisionTreeClassifier, RandomForestClassifier,
                               GradientBoostingClassifier, KNeighborsClassifier, LinearSVC]


@dataclass
class ModelExperimentTagParam:
    model_tag: experiment_model_tags
    model_param: str | None


@dataclass
class ModelExperimentBooter:
    model_tag_param: list[ModelExperimentTagParam]
    preproc_params: PreprocessingParameters | None
    features_src: pathlib.Path
    labels_src: pathlib.Path
    ds_src_is_preproc: bool


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
    f1_scores: list[float]

    def __repr__(self):
        if len(self.f1_scores) > 1:
            out = "configuration: {}, performance: avg {}, min {}, max {}".format(self.config, round(statistics.mean(self.f1_scores), 4), round(min(self.f1_scores), 4), round(max(self.f1_scores), 4))
        else:
            out = "configuration: {}, performance: {}".format(self.config, round(self.f1_scores[0], 4))
        return out


"""
    Model exploitation
"""


@dataclass
class ModelExploitationBooter:
    preproc_params: PreprocessingParameters | None
    dataset_src: pathlib.Path
    id_src: pathlib.Path | None
    ds_src_is_preproc: bool
