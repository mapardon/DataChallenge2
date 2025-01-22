from Structs import ModelExperimentTagParam, ModelExperimentBooter, PreprocessingParameters

from MachineLearningProcedure import MachineLearningProcedure

if __name__ == '__main__':

    preproc_pars = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector=None, remove_uninformative_features=True, remove_correlated_features=True, feature_selector=None)
    mi_models: list[ModelExperimentBooter] = [ModelExperimentBooter(ModelExperimentTagParam("lm", None), preproc_pars), ModelExperimentBooter(ModelExperimentTagParam("dtree", None), preproc_pars),
                                              ModelExperimentBooter(ModelExperimentTagParam("gbc", "n_estimators"), preproc_pars), ModelExperimentBooter(ModelExperimentTagParam("gbc", "subsample"), preproc_pars),
                                              ModelExperimentBooter(ModelExperimentTagParam("gbc", "min_sample_split"), preproc_pars), ModelExperimentBooter(ModelExperimentTagParam("gbc", "max_depth"), preproc_pars)][:1]

    #MachineLearningProcedure(preproc_pars, mi_models).main(["PPI"])

    MachineLearningProcedure(preproc_pars, mi_models).main(["PPI", "PI", "SI", "ME"][:-1])
