from abc import ABC

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class ModelIdentification(ABC):
    """
        Base class for model identification and model selection phases
    """
    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, test_features: pd.DataFrame,
                 test_labels: pd.DataFrame, cv_folds: int, verbose=False):
        self.train_features = train_features
        self.train_labels = train_labels

        self.test_features = test_features
        self.test_labels = test_labels

        self.cv_folds = cv_folds
        self.verbose = verbose

    def parametric_identification(self, model, is_reg_model: bool = False) -> list:
        """
            Split train set in train/test sets to train/test the argument model on these and return the cross validation
            performance of the model.

            :returns: F1 performance
        """

        n_rows_fold = len(self.train_features) // self.cv_folds
        f1 = list()

        X_i = self.train_features
        y_i = self.train_labels

        for i in range(self.cv_folds):
            X_i_tr = pd.concat([X_i.iloc[: n_rows_fold * i], X_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            X_i_ts = X_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            y_i_tr = pd.concat([y_i.iloc[: n_rows_fold * i], y_i.iloc[n_rows_fold * (i + 1):]], axis=0,
                               ignore_index=True)
            y_i_ts = y_i.iloc[n_rows_fold * i: n_rows_fold * (i + 1)]

            # fit and predict
            model.fit(X_i_tr, y_i_tr)
            y_i_pred = np.clip(np.round(model.predict(X_i_ts)).astype(int), 1, 3) if is_reg_model else model.predict(X_i_ts).astype(int)

            # performance
            f1.append(f1_score(y_i_ts, y_i_pred, average='micro'))

        return f1
