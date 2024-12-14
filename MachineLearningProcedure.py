from typing import Literal

from sklearn.linear_model import LinearRegression

from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing
from ParametricIdentification import ParametricIdentification


class MachineLearningProcedure:
    def __init__(self):
        self.preproc_params = list()

    def main(self, modes : Literal["PI", "MI", "ME"]):
        if "PI" in modes:
            self.preprocessing_identification()

        if "MI" in modes:
            self.model_identification()

        if "ME" in modes:
            self.model_exploitation()

    def preprocessing_identification(self):
        dl = DataLoading()
        dl.load_data("data/train_features_short.csv", "data/train_labels_short.csv", False)
        train_features, train_labels = dl.get_train_dataset()

        dp = DataPreprocessing(train_features, train_labels, None)
        dp.numerize_categorical_features("remove")
        dp.features_scaling("minmax")
        train_features, train_labels, validation_features, validation_labels = dp.get_train_validation_datasets()

    def model_identification(self):
        dl = DataLoading()
        dl.load_data("data/train_features_short.csv", "data/train_labels_short.csv", False)
        train_features, train_labels = dl.get_train_dataset()
        train_features, train_labels, validation_features, validation_labels = dp.get_train_validation_datasets()

        pi = ParametricIdentification(train_features, train_labels, validation_features, validation_labels, 5)
        m = LinearRegression()
        print(pi.parametric_identification(m, True))

    def model_exploitation(self):
        pass


if __name__ == '__main__':

    MachineLearningProcedure()
