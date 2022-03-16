from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_generic_rf_parameters: Dict[str, Dict[str, Any]] = {
 "bootstrap": {
  "defaultValue": True,
  "description": "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.",
  "name": "bootstrap",
  "type": "boolean"
 },
 "max_depth": {
  "defaultValue": None,
  "description": "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
  "name": "max_depth",
  "type": "int"
 },
 "max_features_method": {
  "defaultValue": "auto",
  "description": "The number of features to consider when looking for the best split between trees: 'auto': max_features=n_features, 'sqrt': max_features=sqrt(n_features), 'log2': max_features=log2(n_features).",
  "name": "max_features_method",
  "type": "enum[auto,sqrt,log2]"
 },
 "max_samples": {
  "defaultValue": None,
  "description": "If bootstrap is True, the number of samples to draw from X to train each base estimator.",
  "name": "max_samples",
  "type": "float"
 },
 "min_samples_leaf": {
  "defaultValue": 1,
  "description": "The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least `min_samples_leaf` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
  "name": "min_samples_leaf",
  "type": "int"
 },
 "min_samples_split": {
  "defaultValue": 2,
  "description": "The minimum number of samples required to split an internal node.",
  "name": "min_samples_split",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of jobs to run in parallel. `-1` means using all processors",
  "name": "n_jobs",
  "type": "int"
 },
 "n_trees": {
  "defaultValue": 100,
  "description": "The number of trees in the forest.",
  "name": "n_trees",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seeds the randomness of the bootstrapping and the sampling of the features.",
  "name": "random_state",
  "type": "int"
 },
 "train_window_size": {
  "defaultValue": 50,
  "description": "Size of the training windows. Always predicts a single point!",
  "name": "train_window_size",
  "type": "int"
 },
 "verbose": {
  "defaultValue": 0,
  "description": "Controls logging verbosity.",
  "name": "verbose",
  "type": "int"
 }
}


def generic_rf(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Random Forest Regressor (RR)",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/generic_rf",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_generic_rf_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
