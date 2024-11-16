import pandas as pd


class DataLoading:
    def __init__(self, features_src: str, labels_src: str | None, challenge: bool = False, shuffle: bool = False, short: bool = False):
        self.short = short
        self.features: pd.DataFrame = pd.read_csv(features_src)
        self.labels: pd.DataFrame | None = None if challenge else pd.read_csv(labels_src)
        self.data_id: pd.DataFrame | None = None

        if shuffle:
            self.shuffle_dataset()

        if self.short:
            self.features = self.features[:500]
            if not challenge:
                self.labels = self.labels[:500]

        self.data_id = self.features.pop("building_id")

    def shuffle_dataset(self):
        ds = pd.concat([self.features, self.labels[[self.labels.columns.to_list()[-1]]]], axis="columns")
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split features/labels and remove id
        self.labels = ds[ds.columns.to_list()[-1]]
        self.features = ds[ds.columns.to_list()[:-1]]

    def get_train_dataset(self):
        return self.features, self.labels

    def get_challenge_dataset(self):
        return self.features, self.data_id


if __name__ == '__main__':
    features = "data/train_features.csv"
    labels = "data/train_labels.csv"
    dl = DataLoading(features, labels, False, False, False)
    print(len(dl.get_dataset()[0]))
