from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    preproc_pars = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector=None, remove_uninformative_features=True, feature_selection=None)
    mi_models = ["lm", "dtree"]

    #MachineLearningProcedure(preproc_pars, mi_models).main(["PPI"])

    MachineLearningProcedure(preproc_pars, mi_models).main(["PPI", "PI", "SI", "ME"])
