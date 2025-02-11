import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ParametricIdentificationCV import ParametricIdentificationCV
from Structs import ModelExperimentTagParam, ModelExperimentConfiguration, ModelExperimentResult


class ParametricIdentification(ParametricIdentificationCV):
    """
        Test by cross-validation the performance of different models with different parameters.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, cv_folds: int):
        super().__init__(train_features, train_labels, cv_folds)

    def parametric_identification(self, config: ModelExperimentTagParam) -> list[ModelExperimentResult]:
        if config.model_param is None:
            {"lm": self.lm, "dtree": self.dtree, "rforest": self.rforest, "knn": self.knn, "svm": self.svm}[config.model_tag]()
        else:
            {"gbc": self.gbc}[config.model_tag](config.model_param)

        return self.candidates

    def lm(self):
        lm = LinearRegression()
        f1 = self.parametric_identification_cv(lm, True)
        self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "lm", lm, True, None), f1))

    def dtree(self):
        for c in ["entropy", "gini", "log_loss"]:
            for s in ["best", "random"]:
                dtree = DecisionTreeClassifier(criterion=c, splitter=s)
                f1 = self.parametric_identification_cv(dtree, False)
                self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "dtree", dtree, False, {"criterion": c, "splitter": s}), f1))

    def rforest(self):
        for n_est in [10, 50, 100, 200, 300]:
            # /!\ Large forests can make the pipelining of MachineLearningProcedure fail
            rforest = RandomForestClassifier(n_estimators=n_est, criterion="log_loss")
            f1 = self.parametric_identification_cv(rforest, False)
            self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "rforest", rforest, False, {"n_estimators": n_est, "criterion": "log_loss"}), f1))

    def knn(self):
        for nn in [2, 5, 10]:
            knc = KNeighborsClassifier(nn)
            f1 = self.parametric_identification_cv(knc, False)
            self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "knn", knc, False, {"n_neighbors": nn}), f1))

    def svm(self):
        svm = LinearSVC()
        f1 = self.parametric_identification_cv(svm, False)
        self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "svm", svm, False, None), f1))

    """
    neural networks
    ensembling
    """

    def gbc(self, par="n_estimators"):
        """ :param par: possible values: n_estimators, subsample, min_sample_split, max_depth """

        if par == "n_estimators":
            for n in [50, 100, 200, 300, 400, 500]:
                gbc = GradientBoostingClassifier(loss="log_loss", n_estimators=n)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "gbc", gbc, False, {"n_estimators": n}), f1))

        elif par == "subsample":
            for s in [0.1, 0.5, 0.75, 0.9, 1.0]:
                gbc = GradientBoostingClassifier(loss="log_loss", subsample=s)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "gbc", gbc, False, {"subsample": s}), f1))

        elif par == "min_sample_split":
            for mss in [2, 3, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", min_samples_split=mss)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "gbc", gbc, False, {"min_sample_split": mss}), f1))

        elif par == "max_depth":
            for md in [2, 3, 4, 5, 10]:
                gbc = GradientBoostingClassifier(loss="log_loss", max_depth=md)
                f1 = self.parametric_identification_cv(gbc, False)
                self.candidates.append(ModelExperimentResult(ModelExperimentConfiguration("PI", "gbc", gbc, False, {"max_depth": md}), f1))
