from sklearn.linear_model import LinearRegression

from DataLoading import DataLoading
from DataPreprocessing import DataPreprocessing
from ParametricIdentification import ParametricIdentification


class MachineLearningProcedure:
    def __init__(self):
        self.main()

    def main(self):
        dl = DataLoading()
        dl.load_data("data/train_features_short.csv", "data/train_labels_short.csv", False)
        train_features, train_labels = dl.get_train_dataset()

        dp = DataPreprocessing(train_features, train_labels, None)
        dp.numerize_categorical_features("remove")
        dp.features_scaling("minmax")
        train_features, train_labels, validation_features, validation_labels = dp.get_train_validation_datasets()

        pi = ParametricIdentification(train_features, train_labels, validation_features, validation_labels, 5)
        m = LinearRegression()
        print(pi.parametric_identification(m, True))


if __name__ == '__main__':

    MachineLearningProcedure()
