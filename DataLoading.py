import pandas as pd


class DataLoading:
    def __init__(self, features_src, labels_src, challenge=False, shuffle=False, short=False):
        self.features: pd.DataFrame = pd.read_csv(features_src)
        self.labels: pd.DataFrame | None = None if challenge else pd.read_csv(labels_src)
        self.short = short

        if shuffle:
            self.shuffle_dataset()

        if self.short:
            self.features = self.features[:500]
            if self.labels is not None:
                self.labels = self.labels[:500]

    def shuffle_dataset(self):
        ds = pd.concat([self.features, self.labels[[self.labels.columns.to_list()[-1]]]], axis="columns")
        ds = ds.sample(frac=1)
        ds.reset_index(inplace=True, drop=True)

        # split features/labels and remove id
        self.labels = ds[ds.columns.to_list()[-1]]
        self.features = ds[ds.columns.to_list()[:-1]]
        if "building_id" in self.features:
            self.features.pop("building_id")

    def get_dataset(self):
        return self.features, self.labels


if __name__ == '__main__':
    features = "data/train_features.csv"
    labels = "data/train_labels.csv"
    dl = DataLoading(features, labels, False, False, False)
    print(len(dl.get_dataset()[0]))
