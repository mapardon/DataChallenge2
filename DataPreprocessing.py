from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


@dataclass
class PreprocessingParameters:
    numerizer: Literal["remove", "one-hot"] | None = None
    scaler: Literal["minmax"] | None = None
    outlier_detector: Literal[""] | None = None
    remove_uninformative_features: bool = False
    feature_selection: Literal["RFE"] | list | np.ndarray | None = None

    def __repr__(self):
        # TODO: all parameters
        return "Preprocessing params: numerizer: {}, scaler: {}".format(self.numerizer, self.scaler)


@dataclass
class PreprocessingOutput:
    outliers_detection_res: int | None = None
    uninformative_features: list[str] | None = None
    selected_features: str | None = None


class DataPreprocessing:
    """
        Perform the requested preprocessing operations on provided dataset.
    """

    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame | None, data_id: pd.DataFrame | None):

        if labels is None and data_id is None:
            raise Warning("Both label and data_id have been set to None for preprocessing.")

        self.features: pd.DataFrame | None = features
        self.labels: pd.DataFrame | None = labels
        self.data_id: pd.DataFrame | None = data_id

    def preprocessing(self, parameters: PreprocessingParameters) -> PreprocessingOutput:
        pars = parameters
        res = PreprocessingOutput()

        if pars.numerizer is not None:
            self.numerize_categorical_features(pars.numerizer)

        if pars.scaler is not None:
            self.features_scaling(pars.scaler)

        if pars.outlier_detector is not None:
            res.outliers_detection_res = self.outlier_detection(pars.outlier_detector)

        if pars.remove_uninformative_features:
            res.uninformative_features = self.remove_uninformative_features()

        if pars.feature_selection is not None:
            res.selected_features = self.feature_selection(feat_selector=pars.feature_selection)

        return  res

    # Retrieve preprocessed datasets

    def get_train_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.features, self.labels

    def get_train_validation_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.features.iloc[:round(len(self.features) * 0.75), :], self.labels[:round(len(self.features) * 0.75)],
                self.features.iloc[round(len(self.features) * 0.75):, :], self.labels[round(len(self.features) * 0.75):])

    def get_challenge_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.features, self.data_id

    # Preprocessing operations

    def numerize_categorical_features(self, numerizer: Literal["remove", "one-hot"]) -> None:
        if numerizer == "remove":
            self.features = self.features.select_dtypes([np.number])
        
        elif numerizer == "one-hot":
            num_features = self.features.select_dtypes(["number"])
            obj_features = self.features.select_dtypes(["object"])
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)
            encoder.set_output(transform="pandas")
            obj_features = encoder.fit_transform(obj_features)
            self.features = pd.concat([num_features, obj_features], axis="columns")

    def features_scaling(self, scaler: Literal["minmax"]) -> None:
        if scaler == "minmax":
            scaler = MinMaxScaler()
            self.features[self.features.select_dtypes([np.number]).columns] = scaler.fit_transform(self.features.select_dtypes([np.number]))

    def outlier_detection(self, outlier_detector: Literal[""]):
        pass

    def remove_uninformative_features(self):
        """
            Remove features having at least 95% of observations with same value (probably not useful in the model).
        """

        n_features = len(self.features)
        uninformative_threshold = 0.95
        modes = self.features.mode()
        uninformative_features = list()

        for i, m in enumerate(modes):
            if self.features[self.features[m] == modes.iloc[0, i]].shape[0] / n_features >= uninformative_threshold:
                uninformative_features.append(m)

        self.features = self.features.drop(columns=uninformative_features)
        return uninformative_features

    def feature_selection(self, feat_selector: Literal["RFE"] | list | np.ndarray | None):
        pass
