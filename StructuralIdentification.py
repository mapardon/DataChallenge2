import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ParametricIdentificationCV import ModelExperimentResult, ModelExperimentConfiguration


class StructuralIdentification:
    """
        Compare the models having shown the best performance during parametric identification. Use the train set and
        test on validation set unused during parametric identification.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, validation_features: pd.DataFrame,
                 validation_labels: pd.DataFrame):

        self.train_features: pd.DataFrame = train_features
        self.train_labels: pd.DataFrame = train_labels
        self.validation_features: pd.DataFrame = validation_features
        self.validation_labels: pd.DataFrame = validation_labels
        self.candidates: list[ModelExperimentResult] = list()

    def model_selection(self, pi_candidates: list[ModelExperimentResult]):
        for c in pi_candidates:
            m = c.config.model
            m.fit(self.train_features, self.train_labels.values.ravel())
            y_i_vs_pred = np.clip(np.round(m.predict(self.validation_features)).astype(int), 1, 3) if c.config.is_reg_model else m.predict(self.validation_features).astype(int)
            f1 = f1_score(self.validation_labels, y_i_vs_pred, average='micro')
            self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("SI", c.config.model_tag, m, c.config.is_reg_model, c.config.model_params), [f1]))

        return sorted(self.candidates, reverse=True, key=lambda x: x.f1)
