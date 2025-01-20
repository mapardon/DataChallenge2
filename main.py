from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure
from ParametricIdentification import experiment_models

if __name__ == '__main__':

    preproc_pars = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector=None, remove_uninformative_features=True, remove_correlated_features=True, feature_selector=None)
    mi_models: list[experiment_models] = ["lm", "dtree"][:1]

    MachineLearningProcedure(preproc_pars, mi_models).main(["PPI"])

    #MachineLearningProcedure(preproc_pars, mi_models).main(["PPI", "PI", "SI", "ME"])
