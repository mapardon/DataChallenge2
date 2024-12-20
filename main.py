from DataAnalysis import DataAnalysis
from DataPreprocessing import PreprocessingParameters
from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    #DataAnalysis().exploratory_analysis()
    MachineLearningProcedure(PreprocessingParameters("remove", "minmax")).main(["PI"])
