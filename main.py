from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    preproc_pars = PreprocessingParameters("remove", "minmax")
    mi_models = ["lm", "dtree"][:1]

    MachineLearningProcedure(preproc_pars, mi_models).main(["PI", "SI", "ME"])

    # MachineLearningProcedure(None, mi_models).main(["PPI", "PI", "SI", "ME"])
