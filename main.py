from Structs import ModelExperimentTagParam, ModelExperimentBooter, PreprocessingParameters

from MachineLearningProcedure import MachineLearningProcedure

if __name__ == '__main__':

    preproc_pars = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector=None,
                                           remove_uninformative_features=True, remove_correlated_features=True,
                                           feature_selector=None)
    model_tags_params: list[ModelExperimentTagParam] = [ModelExperimentTagParam("lm", None),
                                                        ModelExperimentTagParam("dtree", None),
                                                        ModelExperimentTagParam("gbc", "n_estimators"),
                                                        ModelExperimentTagParam("gbc", "subsample"),
                                                        ModelExperimentTagParam("gbc", "min_sample_split"),
                                                        ModelExperimentTagParam("gbc", "max_depth")]

    model_tags_params = model_tags_params[:2]
    mi_configs: ModelExperimentBooter = ModelExperimentBooter(model_tags_params, preproc_pars, None)

    print(mi_configs)

    MachineLearningProcedure(None, mi_configs).main(["PI", "SI", "ME"])
