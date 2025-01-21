import statistics

import pandas as pd
from sklearn.linear_model import LinearRegression

from DataPreprocessing import PreprocessingParameters, DataPreprocessing, PreprocessingOutput
from ParametricIdentificationCV import ParametricIdentificationCV


class PreprocExperimentResult:
    def __init__(self, configuration: PreprocessingParameters, performance: list[float], preprocessing_output: list[PreprocessingOutput]):
        self.configuration: PreprocessingParameters = configuration
        self.f1_scores: list[float] = performance
        self.preprocessing_output: list[PreprocessingOutput] = preprocessing_output

    def __repr__(self):
        out = "\nPreprocessing Experiment\n {}\n performance: {} ()".format(self.configuration, round(statistics.mean(self.f1_scores), 4), round(statistics.stdev(self.f1_scores), 4))

        if self.configuration.outlier_detector is not None:
            out += "\n n_outliers_removed: {}-{}".format(min(self.preprocessing_output, key=lambda x: x.outliers_detection_res).outliers_detection_res, max(self.preprocessing_output, key=lambda x: x.outliers_detection_res).outliers_detection_res)

        if self.configuration.remove_uninformative_features:
            out += "\n uninformative features removed: {}".format(", ".join(["{} ({})".format(feat, [feat for res in self.preprocessing_output for feat in res.uninformative_features].count(feat)) for feat in set([feat for res in self.preprocessing_output for feat in res.uninformative_features])]))

        if self.configuration.remove_correlated_features:
            out += "\n correlated features removed: {}".format(", ".join(["{} ({})".format(feat, [feat for res in self.preprocessing_output for feat in res.correlated_features].count(feat)) for feat in set([feat for res in self.preprocessing_output for feat in res.correlated_features])]))

        if self.configuration.feature_selector is not None:
            out += "\n features selected: {}".format(", ".join(["{} ({})".format(feat, [feat for res in self.preprocessing_output for feat in res.selected_features].count(feat)) for feat in set([feat for res in self.preprocessing_output for feat in res.selected_features])]))

        return out


class PreprocessingIdentification:
    """
        Test by cross-validation the performance of different preprocessing parameters. All preprocessing configurations
        are tested with a linear model (not optimal but acceptable approximation considering computing restrictions).
    """

    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame, cv_folds: int = 5):
        self.n_experiments: int = cv_folds

        self.features: pd.DataFrame = features
        self.labels: pd.DataFrame = labels

    def preprocessing_identification(self, configuration: PreprocessingParameters) -> PreprocExperimentResult:
        """
            :returns: PreprocExperimentResult object with tested configuration and performance of experiments
        """

        lm = LinearRegression()
        ppe_res = PreprocExperimentResult(configuration, list(), list())

        for _ in range(self.n_experiments):
            features = self.features.copy(deep=True)
            labels = self.labels.copy(deep=True)

            dp = DataPreprocessing(features, labels, None)
            preprocessing_output = dp.preprocessing(configuration)
            features, labels = dp.get_train_dataset()

            f1_scores = ParametricIdentificationCV(features, labels, 2).parametric_identification_cv(lm, True)

            ppe_res.preprocessing_output.append(preprocessing_output)
            ppe_res.f1_scores += f1_scores

        return ppe_res
