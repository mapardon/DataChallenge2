import statistics

import pandas as pd
from sklearn.linear_model import LinearRegression

from DataPreprocessing import PreprocessingParameters, DataPreprocessing
from ParametricIdentificationCV import ParametricIdentificationCV


class PreprocExperimentResult:
    def __init__(self, configuration: PreprocessingParameters, performance: list[float]):
        self.configuration: PreprocessingParameters = configuration
        self.f1_scores: list[float] = performance
        # TODO some experiments will require extra outputs

    def __repr__(self):
        return "Preprocessing Experiment\n {}\n performance: {}".format(self.configuration, round(statistics.mean(self.f1_scores), 4))


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
        f1_scores: list[float] = list()

        for _ in range(self.n_experiments):
            features = self.features.copy(deep=True)
            labels = self.labels.copy(deep=True)

            dp = DataPreprocessing(features, labels, None)
            dp.preprocessing(configuration)
            features, labels, _, _ = dp.get_train_validation_datasets()

            # TODO: some experiments will require extra outputs

            f1_scores = ParametricIdentificationCV(features, labels, 2).parametric_identification_cv(lm, True)

        return PreprocExperimentResult(configuration, f1_scores)
