import os
import pathlib

import pandas as pd


class DataLoading:
    def __init__(self):
        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.data_id: pd.Series | None = None

    def load_data(self, features_src: os.PathLike, labels_src: os.PathLike | None, shuffle: bool = False):
        self.features = pd.read_csv(features_src)
        if labels_src is not None:
            self.labels = pd.read_csv(labels_src)

            if shuffle:
                self.shuffle_dataset()

            if "building_id" in self.features:
                _ = self.labels.pop("building_id")

        if "building_id" in self.features:
            self.data_id = self.features.pop("building_id")

    def shuffle_dataset(self):
        ds = pd.concat([self.features, self.labels[self.labels.columns.to_list()[-1]]], axis="columns")  # drop building_id in labels to avoid same column name after concat
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split features/labels
        self.labels = ds[ds.columns.to_list()[-1:]]
        self.features = ds[ds.columns.to_list()[:-1]]

    def get_train_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.labels is None:
            raise Warning("'self.labels' object is None, did you want to call 'self.get_challenge_dataset()'?")
        return self.features, self.labels

    def get_challenge_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.features, self.data_id


if __name__ == '__main__':
    features: os.PathLike = pathlib.Path("data/train_features_short.csv")
    labels: os.PathLike = pathlib.Path("data/train_labels_short.csv")
    challenge: os.PathLike = pathlib.Path("data/challenge_features.csv")

    dl = DataLoading()
    dl.load_data(features, labels, False)
    print("a", dl.get_train_dataset())

    dl.load_data(features, labels, True)
    print("b", dl.get_train_dataset())

    dl.load_data(challenge, None, False)
    print("c", dl.get_challenge_dataset())
