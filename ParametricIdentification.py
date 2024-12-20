from typing import Literal

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from ModelIdentification import ModelIdentification


class ExperimentResultPI:
    def __init__(self, model_tag: Literal["lm", "dtree"], model, is_reg_model: bool, performance: list[float], param: dict | None):
        self.model_tag: Literal["lm", "dtree"] = model_tag
        self.model = model
        self.is_reg_model: bool = is_reg_model
        self.f1: list[float] = performance
        self.param: dict | None = param

    def __repr__(self):
        return "{} ({}): {}".format(self.model_tag, self.param, self.f1)


class ParametricIdentification(ModelIdentification):
    """
        Test by cross-validation the performance of different models with different parameters.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, validation_features: pd.DataFrame,
                 validation_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, validation_features, validation_labels, cv_folds)

        self.candidates: list[ExperimentResultPI] = list()

    def parametric_identification(self, model_tags: list[Literal["lm", "tree"]]) -> list[ExperimentResultPI]:
        for model_tag in model_tags:
            {"lm": self.lm, "tree": self.tree}[model_tag]()

        return self.candidates

    def lm(self):
        lm = LinearRegression()
        f1 = self.parametric_identification_cv(lm, True)
        self.candidates.append(ExperimentResultPI("lm", lm, True, f1, None))

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                f1 = self.parametric_identification_cv(dtree, False)
                self.candidates.append(ExperimentResultPI("dtree", dtree, False, f1, {"criterion": c, "splitter": s}))
