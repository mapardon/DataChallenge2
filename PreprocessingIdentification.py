import statistics

import pandas as pd
from sklearn.linear_model import LinearRegression

from DataPreprocessing import PreprocessingParameters, DataPreprocessing
from ParametricIdentificationCV import ParametricIdentificationCV


class PreprocExperimentResult:
    def __init__(self, configuration: PreprocessingParameters, performance: list[float]):
        self.configuration: PreprocessingParameters = configuration
        self.performance: list[float] = performance
        # TODo some experiments will require extra outputs

    def __repr__(self):
        return "Preprocessing Experiment\n {}\n performance: {}".format(self.configuration, round(statistics.mean(self.performance), 4))


class PreprocessingIdentification:
    """
        Test by cross-validation the performance of different preprocessing parameters. All preprocessing configurations
        are tested with a linear model (not optimal but acceptable approximation considering computing restrictions).
    """

    def __init__(self, configurations: list[PreprocessingParameters], features: pd.DataFrame, labels: pd.DataFrame, cv_folds: int = 5):
        self.configurations: list[PreprocessingParameters] = configurations
        self.cv_folds: int = cv_folds
        self.candidates: list[PreprocExperimentResult] = list()

        self.features: pd.DataFrame = features
        self.labels: pd.DataFrame = labels

    def preprocessing_identification(self) -> list[PreprocExperimentResult]:
        """
            :returns: preprocessing experiment results sorted by decreasing average performance
        """

        lm = LinearRegression()
        for conf in self.configurations:
            self.candidates.append(PreprocExperimentResult(conf, list()))

            for _ in range(self.cv_folds):
                features = self.features.copy(deep=True)
                labels = self.labels.copy(deep=True)

                dp = DataPreprocessing(features, labels, None)
                dp.preprocessing(conf)
                features, labels, _, _ = dp.get_train_validation_datasets()

                # TODO: some experiments will require extra outputs

                perf = ParametricIdentificationCV(features, labels, 2, False).parametric_identification_cv(lm, True)
                self.candidates[-1].performance.extend(perf)

        return sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x.performance))
