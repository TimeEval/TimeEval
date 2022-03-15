from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_generic_xgb_parameters: Dict[str, Dict[str, Any]] = {
 "booster": {
  "defaultValue": "gbtree",
  "description": "Booster to use",
  "name": "booster",
  "type": "enum[gbtree,gblinear,dart]"
 },
 "colsample_bylevel": {
  "defaultValue": None,
  "description": "Subsample ratio of columns for each level.",
  "name": "colsample_bylevel",
  "type": "float"
 },
 "colsample_bynode": {
  "defaultValue": None,
  "description": "Subsample ratio of columns for each split.",
  "name": "colsample_bynode",
  "type": "float"
 },
 "colsample_bytree": {
  "defaultValue": None,
  "description": "Subsample ratio of columns when constructing each tree.",
  "name": "colsample_bytree",
  "type": "float"
 },
 "learning_rate": {
  "defaultValue": 0.1,
  "description": "Boosting learning rate (xgb\u2019s `eta`)",
  "name": "learning_rate",
  "type": "float"
 },
 "max_depth": {
  "defaultValue": None,
  "description": "Maximum tree depth for base learners.",
  "name": "max_depth",
  "type": "int"
 },
 "max_samples": {
  "defaultValue": None,
  "description": "Subsample ratio of the training instance.",
  "name": "max_samples",
  "type": "float"
 },
 "n_estimators": {
  "defaultValue": 100,
  "description": "Number of gradient boosted trees. Equivalent to number of boosting rounds.",
  "name": "n_estimators",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of jobs to run in parallel. `-1` means using all processors.",
  "name": "n_jobs",
  "type": "int"
 },
 "n_trees": {
  "defaultValue": 1,
  "description": "If >1, then boosting random forests with `n_trees` trees.",
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
 "tree_method": {
  "defaultValue": "auto",
  "description": "Tree method to use. Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available. `exact` is slowest, `hist` is fastest. Prefer `hist` and `approx` over `exact`, because for most datasets they have comparative quality, but are significantly faster.",
  "name": "tree_method",
  "type": "enum[auto,exact,approx,hist]"
 },
 "verbose": {
  "defaultValue": 0,
  "description": "Controls logging verbosity.",
  "name": "verbose",
  "type": "int"
 }
}


def generic_xgb(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="XGBoosting (RR)",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/generic_xgb",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_generic_xgb_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
