"""
    Operations not in the scope of ML procedure or model identification
"""
import pathlib

from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing
from Structs import PreprocessingParameters


def create_preprocessed_datafiles(ppars: PreprocessingParameters):
    """
        Apply specified preprocessing parameters to training & challenge data and store it on disk.
    """

    # Training data
    dl = DataLoading()
    dl.load_train_data(pathlib.Path("data/train_features.csv"), pathlib.Path("data/train_labels.csv"), shuffle=True)
    tr_features, tr_labels = dl.get_train_dataset()

    dp = DataPreprocessing(tr_features, tr_labels, None)
    dp.preprocessing(ppars)
    tr_features, tr_labels = dp.get_train_dataset()

    # challenge data
    dl = DataLoading()
    dl.load_challenge_data(pathlib.Path("data/challenge_features.csv"), None, False)
    ch_features, ch_id = dl.get_challenge_dataset()

    dp = DataPreprocessing(ch_features, None, ch_id)
    dp.preprocessing(ppars)
    ch_features, ch_id = dp.get_challenge_dataset()

    # store in .csv
    tr_features.to_csv(open("data/train_features_preprocessed.csv", 'w', encoding='utf-8'), index=False, lineterminator="\n")
    tr_labels.to_csv(open("data/train_labels_preprocessed.csv", 'w', encoding='utf-8'), index=False, lineterminator="\n")
    ch_features.to_csv(open("data/challenge_features_preprocessed.csv", 'w', encoding='utf-8'), index=False, lineterminator="\n")
    ch_id.to_csv(open("data/challenge_id_preprocessed.csv", 'w', encoding='utf-8'), index=False, lineterminator="\n")


if __name__ == '__main__':

    ppars = PreprocessingParameters("one-hot", "minmax", None, int(), False, False, None, float())
    create_preprocessed_datafiles(ppars)
