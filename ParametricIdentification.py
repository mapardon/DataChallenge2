from typing import Literal

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from ModelIdentification import ModelIdentification, ModelExperimentResult


class ParametricIdentification(ModelIdentification):
    """
        Test by cross-validation the performance of different models with different parameters.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, cv_folds)

        self.candidates: list[ModelExperimentResult] = list()

    def parametric_identification(self, model_tags: list[Literal["lm", "tree"]]) -> list[ModelExperimentResult]:
        for model_tag in model_tags:
            {"lm": self.lm, "tree": self.tree}[model_tag]()

        return self.candidates

    def lm(self):
        lm = LinearRegression()
        f1 = self.parametric_identification_cv(lm, True)
        self.candidates.append(ModelExperimentResult("PI", "lm", lm, True, None, f1))

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                f1 = self.parametric_identification_cv(dtree, False)
                self.candidates.append(ModelExperimentResult("PI", "dtree", dtree, False, {"criterion": c, "splitter": s}, f1))
