from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for Donut
def post_donut(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 120)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_donut_parameters: Dict[str, Dict[str, Any]] = {
 "epochs": {
  "defaultValue": 256,
  "description": "Number of training passes over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "latent_size": {
  "defaultValue": 5,
  "description": "Dimensionality of encoding",
  "name": "latent_size",
  "type": "int"
 },
 "linear_hidden_size": {
  "defaultValue": 100,
  "description": "Size of linear hidden layer",
  "name": "linear_hidden_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "regularization": {
  "defaultValue": 0.001,
  "description": "Factor for regularization in loss",
  "name": "regularization",
  "type": "float"
 },
 "use_column_index": {
  "defaultValue": 0,
  "description": "The column index to use as input for the univariate algorithm for multivariate datasets. The selected single channel of the multivariate time series is analyzed by the algorithms. The index is 0-based and does not include the index-column ('timestamp'). The single channel of an univariate dataset, therefore, has index 0.",
  "name": "use_column_index",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 120,
  "description": "Size of sliding windows",
  "name": "window_size",
  "type": "int"
 }
}


def donut(params: Optional[ParameterConfig] = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Donut",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/donut",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_donut,
        param_schema=_donut_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
