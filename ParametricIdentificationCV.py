import statistics
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

experiment_model_tags = Literal["lm", "dtree", "gbc"]
experiment_model_types = Union[LinearRegression, DecisionTreeClassifier, GradientBoostingClassifier]


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


class ParametricIdentificationCV:
    """
        Base class for model identification phase
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, cv_folds: int):
        self.train_features: pd.DataFrame = train_features
        self.train_labels: pd.DataFrame = train_labels

        self.candidates: list[ModelExperimentResult] = list()
        self.cv_folds: int = cv_folds

    def parametric_identification_cv(self, model, is_reg_model: bool = False) -> list[float]:
        """
            Split train set in train/test sets to train/test the argument model on these and return the cross validation
            performance of the model.

            :returns: performance of the model measured with the F1 score
        """

        n_rows_fold = len(self.train_features) // self.cv_folds
        f1_scores = list()

        X_i = self.train_features
        y_i = self.train_labels

        for i in range(self.cv_folds):
            X_i_tr = pd.concat([X_i.iloc[: n_rows_fold * i], X_i.iloc[n_rows_fold * (i + 1):]], axis=0, ignore_index=True)
            X_i_ts = X_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            y_i_tr = pd.concat([y_i.iloc[: n_rows_fold * i], y_i.iloc[n_rows_fold * (i + 1):]], axis=0, ignore_index=True)
            y_i_ts = y_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            # fit and predict
            model.fit(X_i_tr, y_i_tr.values.ravel())
            y_i_pred = np.clip(np.round(model.predict(X_i_ts)).astype(int), 1, 3) if is_reg_model else model.predict(X_i_ts).astype(int)

            # performance
            f1_scores.append(f1_score(y_i_ts, y_i_pred, average='micro'))

        return f1_scores
