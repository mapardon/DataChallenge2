import statistics

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ModelIdentification import ModelIdentification, ModelExperimentResult


class StructuralIdentification(ModelIdentification):
    """
        Compare the models having shown the best performance during parametric identification. Use the train set and
        test on validation set unused during parametric identification.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, validation_features: pd.DataFrame,
                 validation_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, cv_folds, True)

        self.candidates: list[ModelExperimentResult] = list()

        self.validation_features: pd.DataFrame = validation_features
        self.validation_labels: pd.DataFrame = validation_labels

    def model_selection(self, pi_candidates: list[ModelExperimentResult]):
        for c in pi_candidates:
            m = c.model
            m.fit(self.train_features, self.train_labels)
            y_i_vs_pred = np.clip(np.round(m.predict(self.validation_features)).astype(int), 1, 3) if c.is_reg_model else m.predict(self.validation_features).astype(int)
            f1 = f1_score(self.validation_labels, y_i_vs_pred, average='micro')
            self.candidates.append(ModelExperimentResult("SI", c.model_tag, m, c.is_reg_model, c.model_param, [f1]))

        self.candidates.sort(reverse=True, key=lambda x: x.f1)

        # print results of the testing of the most promising models
        if self.verbose:
            print("\n * MODEL TESTING *")
            print(" -> performance:")
            for c in self.candidates:
                print("{} ({})".format(c.model_tag, c.model_param))
                print("candidate.auc: {}".format(round(statistics.mean(c.f1), 5)))

        return self.candidates[0]
