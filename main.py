from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    preproc_pars = PreprocessingParameters("one-hot", "minmax")
    mi_models = ["lm", "dtree"]

    # MachineLearningProcedure(preproc_pars, mi_models).main(["PI", "SI", "ME"])

    MachineLearningProcedure(None, None).main(["DA"])
