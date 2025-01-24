import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from DataPreprocessing import DataPreprocessing
from Structs import PreprocessingParameters, PreprocExperimentResult
from ParametricIdentificationCV import ParametricIdentificationCV


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
