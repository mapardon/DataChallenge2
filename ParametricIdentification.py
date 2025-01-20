import statistics
from typing import Literal

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from ParametricIdentificationCV import ParametricIdentificationCV, ModelExperimentResult


experiment_models = Literal["lm", "dtree", "gbc"]


class ParametricIdentification(ParametricIdentificationCV):
    """
        Test by cross-validation the performance of different models with different parameters.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, cv_folds)

    def parametric_identification(self, model_tags: list[experiment_models]) -> list[ModelExperimentResult]:
        for model_tag in model_tags:
            {"lm": self.lm, "dtree": self.dtree}[model_tag]()

        return sorted(self.candidates, reverse=True, key=lambda x: statistics.mean(x.f1))

    def lm(self):
        lm = LinearRegression()
        f1 = self.parametric_identification_cv(lm, True)
        self.candidates.append(ModelExperimentResult("PI", "lm", lm, True, None, f1))

    def dtree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                f1 = self.parametric_identification_cv(dtree, False)
                self.candidates.append(ModelExperimentResult("PI", "dtree", dtree, False, {"criterion": c, "splitter": s}, f1))

    def gbc(self, par="n_estimators"):
        """ :param par: possible values: n_estimators, subsample, min_sample_split, max_depth """
        if par == "n_estimators":
            for n in [50, 100, 200, 300, 400, 500]:
                gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=n)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult("gbc", gbc, f1, "n_estimators", n, False, False))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult("gbc", gbc, f1, "subsample", s, False, False))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult("gbc", gbc, f1, "min_sample_split", mss, False, False))

        elif par == "max_depth":
            for max_depth in [2, 3, 4, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=max_depth)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult("gbc", gbc, f1, "max_depth", max_depth, False, False))
