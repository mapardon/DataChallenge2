import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from ModelIdentification import ModelIdentification


class ExperimentResultPi:
    def __init__(self, model_tag, model, is_reg_model, performance, param):
        self.model_tag = model_tag
        self.model = model
        self.is_reg_model = is_reg_model
        self.f1 = performance
        self.param = param


class ParametricIdentification(ModelIdentification):
    """
        Test by cross-validation the performance of different models with different parameters.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, validation_features: pd.DataFrame,
                 validation_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, validation_features, validation_labels, cv_folds)

        self.candidates: list[ExperimentResultPi] = list()

    def parametric_identification(self, models) -> list[ExperimentResultPi]:
        for model in models:
            {"lm": self.lm, "tree": self.tree}[model]()

        return self.candidates

    def lm(self):
        lm = LinearRegression()
        f1 = self.parametric_identification_cv(lm, True)
        self.candidates.append(ExperimentResultPi("lm", lm, True, f1, None))

    def tree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                f1 = self.parametric_identification_cv(dtree, False)
                self.candidates.append(ExperimentResultPi("dtree", dtree, False, f1, {"criterion": c, "splitter": s}))
