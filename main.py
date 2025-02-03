import pathlib
import sys

from Structs import ModelExperimentTagParam, ModelExperimentBooter, PreprocessingParameters

from MachineLearningProcedure import MachineLearningProcedure

SHORT = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True

if SHORT:
    PREPROC_FEATURES_SRC = pathlib.Path("data/train_features_preprocessed_short.csv")
    PREPROC_LABELS_SRC = pathlib.Path("data/train_labels_preprocessed_short.csv")
else:
    PREPROC_FEATURES_SRC = pathlib.Path("data/train_features_preprocessed.csv")
    PREPROC_LABELS_SRC = pathlib.Path("data/train_labels_preprocessed.csv")

if __name__ == '__main__':
    preproc_pars = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector=None,
                                           remove_uninformative_features=False, remove_correlated_features=False,
                                           feature_selector=None)

    preprocessed_features = ['land_surface_condition_n', 'land_surface_condition_o', 'land_surface_condition_t',
                             'foundation_type_r',
                             'roof_type_n', 'roof_type_q', 'roof_type_x', 'ground_floor_type_f', 'ground_floor_type_m',
                             'ground_floor_type_v', 'ground_floor_type_x', 'ground_floor_type_z', 'other_floor_type_j',
                             'other_floor_type_q', 'other_floor_type_s', 'other_floor_type_x', 'position_j',
                             'position_o', 'position_s', 'position_t', 'plan_configuration_a', 'plan_configuration_c',
                             'plan_configuration_d', 'plan_configuration_f', 'plan_configuration_m',
                             'plan_configuration_n', 'plan_configuration_o', 'plan_configuration_q',
                             'plan_configuration_s', 'plan_configuration_u', 'legal_ownership_status_a',
                             'legal_ownership_status_r', 'legal_ownership_status_v', 'legal_ownership_status_w']
    preproc_pars_alt = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector=None,
                                               outlier_detector_nn=int(), remove_uninformative_features=False,
                                               remove_correlated_features=False, feature_selector=preprocessed_features,
                                               feature_selection_prop=1/2)

    model_tags_params: list[ModelExperimentTagParam] = [ModelExperimentTagParam("lm", None),
                                                        ModelExperimentTagParam("dtree", None),
                                                        ModelExperimentTagParam("gbc", "n_estimators"),
                                                        ModelExperimentTagParam("gbc", "subsample"),
                                                        ModelExperimentTagParam("gbc", "min_sample_split"),
                                                        ModelExperimentTagParam("gbc", "max_depth")]
    model_tags_params = [ModelExperimentTagParam("lm", None), ModelExperimentTagParam("knn", None)]

    mi_configs: ModelExperimentBooter = ModelExperimentBooter(model_tags_params, None, None)

    MachineLearningProcedure(preproc_pars, mi_configs).main(["PI", "SI", "ME"])
