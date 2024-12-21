from DataAnalysis import DataAnalysis
from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    # DataAnalysis().exploratory_analysis()

    pi_confs = [PreprocessingParameters("one-hot", None), PreprocessingParameters("remove", None), PreprocessingParameters("remove", "minmax")]
    mi_models = ["lm", "dtree"]
    preproc_pars = PreprocessingParameters("remove", "minmax")
    #MachineLearningProcedure(pi_confs, mi_models, preproc_pars).main(["PPI", "MI"])
    MachineLearningProcedure(pi_confs, mi_models, preproc_pars).main(["PPI"])
