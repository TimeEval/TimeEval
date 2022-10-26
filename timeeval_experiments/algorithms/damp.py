from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for damp
def post_damp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 50)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_damp_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 50,
  "description": "Size of the sliding windows",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "lookahead": {
  "defaultValue": None,
  "description": "Amount of steps to look into the future for deciding which future windows to skip analyzing.",
  "name": "lookahead",
  "type": "int"
 },
 "max_lag": {
  "defaultValue": None,
  "description": "Maximum size to look back in time.",
  "name": "max_lag",
  "type": "int"
 },
 "n_init_train": {
  "defaultValue": 100,
  "description": "Fraction of data used to warmup streaming.",
  "name": "n_init_train",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 }
}


def damp(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DAMP",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/damp",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_damp,
        param_schema=_damp_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
