from abc import ABC
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class ModelExperimentResultOld:
    def __init__(self, model_tag, model, par_tag, par_value, is_reg_model, is_bag=False, estimator_tag=None):
        self.model_tag: str = model_tag
        self.model = model
        self.par_tag: str = par_tag
        self.par_value: str | float | int | None = par_value
        self.is_reg_model: bool = is_reg_model
        self.is_bag: bool = is_bag  # if the method uses an estimator (other than itself)
        self.estimator_tag: str | None = estimator_tag


class ModelExperimentResult:
    def __init__(self, experiment_type: Literal["PI", "SI"], model_tag: Literal["lm", "dtree"], model, is_reg_model: bool,
                 model_param: dict | None, performance: list[float]):
        self.experiment_type: Literal["PI", "SI"] = experiment_type
        self.model_tag: Literal["lm", "dtree"] = model_tag
        self.model = model
        self.is_reg_model: bool = is_reg_model
        self.model_param: dict | None = model_param
        self.f1: list[float] = performance

    def __repr__(self):
        return "{} ({}): {}".format(self.model_tag, self.model_param, self.f1)


class ModelIdentification(ABC):
    """
        Base class for model identification and model selection phases
    """
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, cv_folds: int, verbose=False):
        self.train_features: pd.DataFrame = train_features
        self.train_labels: pd.DataFrame = train_labels

        self.cv_folds: int = cv_folds
        self.verbose: bool = verbose

    def parametric_identification_cv(self, model, is_reg_model: bool = False) -> list:
        """
            Split train set in train/test sets to train/test the argument model on these and return the cross validation
            performance of the model.

            :returns: performance of the model measured with the F1 score
        """

        n_rows_fold = len(self.train_features) // self.cv_folds
        f1 = list()

        X_i = self.train_features
        y_i = self.train_labels

        for i in range(self.cv_folds):
            X_i_tr = pd.concat([X_i.iloc[: n_rows_fold * i], X_i.iloc[n_rows_fold * (i + 1):]], axis=0, ignore_index=True)
            X_i_ts = X_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            y_i_tr = pd.concat([y_i.iloc[: n_rows_fold * i], y_i.iloc[n_rows_fold * (i + 1):]], axis=0, ignore_index=True)
            y_i_ts = y_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            # fit and predict
            model.fit(X_i_tr, y_i_tr)
            y_i_pred = np.clip(np.round(model.predict(X_i_ts)).astype(int), 1, 3) if is_reg_model else model.predict(X_i_ts).astype(int)

            # performance
            f1.append(f1_score(y_i_ts, y_i_pred, average='micro'))

        return f1
