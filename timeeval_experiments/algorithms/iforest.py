from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_iforest_parameters: Dict[str, Dict[str, Any]] = {
 "bootstrap": {
  "defaultValue": "False",
  "description": "If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement is performed.",
  "name": "bootstrap",
  "type": "boolean"
 },
 "max_features": {
  "defaultValue": 1.0,
  "description": "The number of features to draw from X to train each base estimator: `max_features * X.shape[1]`.",
  "name": "max_features",
  "type": "float"
 },
 "max_samples": {
  "defaultValue": None,
  "description": "The number of samples to draw from X to train each base estimator: `max_samples * X.shape[0]`. If unspecified (`None`), then `max_samples=min(256, n_samples)`.",
  "name": "max_samples",
  "type": "float"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of jobs to run in parallel. If -1, then the number of jobs is set to the number of cores.",
  "name": "n_jobs",
  "type": "int"
 },
 "n_trees": {
  "defaultValue": 100,
  "description": "The number of decision trees (base estimators) in the forest (ensemble).",
  "name": "n_trees",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "verbose": {
  "defaultValue": 0,
  "description": "Controls the verbosity of the tree building process logs.",
  "name": "verbose",
  "type": "int"
 }
}


def iforest(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Isolation Forest (iForest)",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/iforest",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_iforest_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
