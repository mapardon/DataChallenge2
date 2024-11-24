import pandas as pd


class DataLoading:
    def __init__(self):
        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.data_id: pd.DataFrame | None = None

    def load_data(self, features_src: str, labels_src: str | None, shuffle: bool = False):
        self.features = pd.read_csv(features_src)
        if labels_src is not None:
            self.labels = pd.read_csv(labels_src)

        if shuffle:
            self.shuffle_dataset()

        self.data_id, _ = self.features.pop("building_id"), self.labels.pop("building_id")

    def shuffle_dataset(self):
        ds = pd.concat([self.features, self.labels[[self.labels.columns.to_list()[-1]]]], axis="columns")
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split features/labels and remove id
        self.labels = ds[ds.columns.to_list()[-1]]
        self.features = ds[ds.columns.to_list()[:-1]]

    def get_train_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.labels is None:
            raise Warning("'self.labels' object is None, did you want to call 'self.get_challenge_dataset()'?")
        return self.features, self.labels

    def get_challenge_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.features, self.data_id


if __name__ == '__main__':
    features = "data/train_features_short.csv"
    challenge = "data/challenge_features.csv"
    labels = "data/train_labels_short.csv"
    dl = DataLoading()
    dl.load_data(challenge, None, False)
    print(dl.get_challenge_dataset())
