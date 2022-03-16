from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_if_lof_parameters: Dict[str, Dict[str, Any]] = {
 "alpha": {
  "defaultValue": 0.5,
  "description": "Scalar that depends on consideration of the dataset and controls the amount of data to be pruned",
  "name": "alpha",
  "type": "float"
 },
 "m": {
  "defaultValue": None,
  "description": "m features with highest scores will be used for pruning",
  "name": "m",
  "type": "int"
 },
 "max_samples": {
  "defaultValue": None,
  "description": "The number of samples to draw from X to train each tree: `max_samples * X.shape[0]`. If unspecified (`None`), then `max_samples=min(256, X.shape[0])`.",
  "name": "max_samples",
  "type": "float"
 },
 "n_neighbors": {
  "defaultValue": 10,
  "description": "Number neighbors to look at in local outlier factor calculation",
  "name": "n_neighbors",
  "type": "int"
 },
 "n_trees": {
  "defaultValue": 200,
  "description": "Number of trees in isolation forest",
  "name": "n_trees",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def if_lof(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="IF-LOF",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/if_lof",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_if_lof_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
