from DataAnalysis import DataAnalysis
from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    # DataAnalysis().exploratory_analysis()

    preproc_pars = PreprocessingParameters("one-hot", "minmax")
    mi_models = ["lm", "dtree"][:1]

    MachineLearningProcedure(preproc_pars, mi_models).main(["PI", "SI", "ME"])

    MachineLearningProcedure(None, mi_models).main(["PPI", "PI", "SI", "ME"])
