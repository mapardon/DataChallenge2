from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class PreprocessingParameters:
    def __init__(self, numerizer: Literal["remove", "one-hot"] | None, scaler: Literal["minmax"] | None):
        self.numerizer: Literal["remove", "one-hot"] | None = numerizer
        self.scaler: Literal["minmax"] | None = scaler

    def __repr__(self):
        return "Preprocessing params: numerizer: {}, scaler: {}".format(self.numerizer, self.scaler)


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

        # TODO keep?
        self.out_detect_res = None
        self.feat_sel_res = None

    def preprocessing(self, parameters: PreprocessingParameters):
        pars = parameters
        if pars.numerizer is not None:
            self.numerize_categorical_features(pars.numerizer)

        if pars.scaler is not None:
            self.features_scaling(pars.scaler)

    # Retrieve preprocessed datasets

    def get_train_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.features, self.labels

    def get_train_validation_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (self.features.iloc[:round(len(self.features) * 0.75), :], self.labels[:round(len(self.features) * 0.75)],
                self.features.iloc[round(len(self.features) * 0.75):, :], self.labels[round(len(self.features) * 0.75):])

    def get_challenge_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.features, self.data_id

    # Preprocessing operations

    def numerize_categorical_features(self, numerizer: Literal["remove", "one-hot"]):
        if numerizer == "remove":
            self.features = self.features.select_dtypes([np.number])
        
        elif numerizer == "one-hot":
            num_features = self.features.select_dtypes(["number"])
            obj_features = self.features.select_dtypes(["object"])
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)
            encoder.set_output(transform="pandas")
            obj_features = encoder.fit_transform(obj_features)
            self.features = pd.concat([num_features, obj_features], axis="columns")

    def features_scaling(self, scaler: Literal["minmax"]):
        if scaler == "minmax":
            scaler = MinMaxScaler()
            self.features[self.features.select_dtypes([np.number]).columns] = scaler.fit_transform(self.features.select_dtypes([np.number]))
