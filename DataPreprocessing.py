import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from DataAnalysis import FEATURES_SRC, LABELS_SRC
from DataLoading import DataLoading

CHALLENGE_SRC = "data/test_features.csv"


class DataPreprocessing:
    def __init__(self, challenge: bool = False, short: bool = False):

        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.test_features: pd.DataFrame | None = None
        self.test_labels: pd.DataFrame | None = None
        self.data_id: pd.DataFrame | None = None

        if not challenge:
            dl = DataLoading(FEATURES_SRC, LABELS_SRC, challenge, not challenge, short)
            self.features, self.labels = dl.get_train_dataset()
        else:
            dl = DataLoading(CHALLENGE_SRC, None, challenge, not challenge, short)
            self.features, self.data_id = dl.get_challenge_dataset()

    def get_train_test_datasets(self):
        train_features = self.features.iloc[:round(len(self.features) * 0.75), :]
        train_labels = self.labels[:round(len(self.features) * 0.75)]
        test_features = self.features.iloc[round(len(self.features) * 0.75):, :]
        test_labels = self.labels[round(len(self.features) * 0.75):]

        return train_features, train_labels, test_features, test_labels

    def get_challenge_dataset(self):
        return self.features, self.data_id

    def features_scaling(self, scaler=None):
        if scaler is not None:
            if scaler == "minmax":
                scaler = MinMaxScaler()
                self.features[self.features.select_dtypes([np.number]).columns] = scaler.fit_transform(self.features.select_dtypes([np.number]))


if __name__ == '__main__':

    dp = DataPreprocessing(False, True)
    print(dp.features.iloc[5])
    dp.features_scaling("minmax")
    print(dp.features.iloc[5])
