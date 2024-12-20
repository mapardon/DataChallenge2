import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ModelIdentification import ModelIdentification
from ParametricIdentification import ExperimentResultPI


class ExperimentResultSi:
    def __init__(self, model_tag, model, is_reg_model, performance, param):
        self.model_tag = model_tag
        self.model = model
        self.is_reg_model = is_reg_model
        self.f1 = performance
        self.param = param


class StructuralIdentification(ModelIdentification):
    """
        Compare the models having shown the best performance during parametric identification. Use the train set and
        validation set not used during parametric identification.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, validation_features: pd.DataFrame,
                 validation_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, validation_features, validation_labels, cv_folds)

        self.candidates: list[ExperimentResultSi] = list()

    def model_selection(self, pi_candidates: list[ExperimentResultPI]):
        for c in pi_candidates:
            m = c.model
            m.fit(self.train_features, self.train_labels)
            y_i_vs_pred = np.clip(np.round(m.predict(self.validation_features)).astype(int), 1, 3) if c.is_reg_model else m.predict(self.validation_features).astype(int)
            f1 = f1_score(self.validation_labels, y_i_vs_pred, average='micro')
            self.candidates.append(ExperimentResultSi(c.model_tag, m, c.is_reg_model, f1, c.param))

        self.candidates.sort(reverse=True, key=lambda x: x.f1)

        # print results of testing of most promising models
        if self.verbose:
            print("\n * MODEL TESTING *")
            print(" -> performance:")
            for c in self.candidates:
                print("{} ({})".format(c.model_tag, c.param))
                print("candidate.auc: {}".format(round(c.f1, 5)))
