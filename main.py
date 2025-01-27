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
                                           remove_uninformative_features=True, remove_correlated_features=True,
                                           feature_selector=None)
    features = ['count_floors_pre_eq', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use',
                'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_school',
                'has_secondary_use_industry', 'has_secondary_use_use_police', 'land_surface_condition_n',
                'land_surface_condition_o', 'land_surface_condition_t', 'foundation_type_h', 'foundation_type_i',
                'foundation_type_r', 'foundation_type_u', 'foundation_type_w', 'roof_type_n', 'roof_type_q',
                'roof_type_x', 'ground_floor_type_f', 'ground_floor_type_m', 'ground_floor_type_v',
                'ground_floor_type_x', 'ground_floor_type_z', 'other_floor_type_j', 'other_floor_type_q',
                'other_floor_type_s', 'other_floor_type_x', 'position_j', 'position_o', 'position_s', 'position_t',
                'plan_configuration_a', 'plan_configuration_c', 'plan_configuration_d', 'plan_configuration_f',
                'plan_configuration_m', 'plan_configuration_n', 'plan_configuration_o', 'plan_configuration_q',
                'plan_configuration_s', 'plan_configuration_u', 'legal_ownership_status_a', 'legal_ownership_status_r',
                'legal_ownership_status_v', 'legal_ownership_status_w']
    preproc_pars_alt = PreprocessingParameters(numerizer="one-hot", scaler="minmax", outlier_detector="lof",
                                               outlier_detector_nn=2, remove_uninformative_features=False,
                                               remove_correlated_features=False, feature_selector=features,
                                               feature_selection_prop=.75)
    model_tags_params: list[ModelExperimentTagParam] = [ModelExperimentTagParam("lm", None),
                                                        ModelExperimentTagParam("dtree", None),
                                                        ModelExperimentTagParam("gbc", "n_estimators"),
                                                        ModelExperimentTagParam("gbc", "subsample"),
                                                        ModelExperimentTagParam("gbc", "min_sample_split"),
                                                        ModelExperimentTagParam("gbc", "max_depth")]

    model_tags_params = model_tags_params[:1] + model_tags_params[2:]

    mi_configs: ModelExperimentBooter = ModelExperimentBooter(model_tags_params, None, (pathlib.Path(PREPROC_FEATURES_SRC),
                                                                                        pathlib.Path(PREPROC_LABELS_SRC)))

    MachineLearningProcedure(preproc_pars_alt, mi_configs).main(["PI", "SI", "ME"])
