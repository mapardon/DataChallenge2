import pathlib

import pandas as pd


class DataLoading:
    def __init__(self):
        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.data_id: pd.Series | None = None

    def load_train_data(self, features_src: pathlib.Path, labels_src: pathlib.Path, shuffle: bool = False):
        self.features = pd.read_csv(features_src)
        self.labels = pd.read_csv(labels_src)

        if shuffle:
            self.shuffle_dataset()

        if "building_id" in self.features:
            _ = self.features.pop("building_id")

        if "building_id" in self.labels:
            _ = self.labels.pop("building_id")

    def shuffle_dataset(self):
        ds = pd.concat([self.features, self.labels[self.labels.columns.to_list()[-1]]], axis="columns")  # drop building_id in labels to avoid same column name after concat
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split features/labels
        self.labels = ds[ds.columns.to_list()[-1:]]
        self.features = ds[ds.columns.to_list()[:-1]]

    def load_challenge_data(self, features_src: pathlib.Path, id_src: pathlib.Path | None, is_preproc: bool):
        self.features = pd.read_csv(features_src)

        if is_preproc:
            self.data_id = pd.read_csv(id_src).pop("building_id")
        else:
            self.data_id = self.features.pop("building_id")

    def get_train_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.labels is None:
            raise Warning("'self.labels' object is None, did you want to call 'self.get_challenge_dataset()'?")
        return self.features, self.labels

    def get_challenge_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.features, self.data_id
