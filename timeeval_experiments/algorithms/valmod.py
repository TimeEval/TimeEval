from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for valmod
def post_valmod(scores: np.ndarray, args: dict) -> np.ndarray:
    window_min = args.get("hyper_params", {}).get("min_anomaly_window_size", 30)
    window_min = max(window_min, 4)
    return ReverseWindowing(window_size=window_min).fit_transform(scores)


_valmod_parameters: Dict[str, Dict[str, Any]] = {
 "exclusion_zone": {
  "defaultValue": 0.5,
  "description": "Size of the exclusion zone as a factor of the window_size. This prevents self-matches.",
  "name": "exclusion_zone",
  "type": "Float"
 },
 "heap_size": {
  "defaultValue": 50,
  "description": "Size of the distance profile heap buffer",
  "name": "heap_size",
  "type": "Int"
 },
 "max_anomaly_window_size": {
  "defaultValue": 40,
  "description": "Maximum sliding window size",
  "name": "max_anomaly_window_size",
  "type": "Int"
 },
 "min_anomaly_window_size": {
  "defaultValue": 30,
  "description": "Minimum sliding window size",
  "name": "min_anomaly_window_size",
  "type": "Int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "Int"
 },
 "verbose": {
  "defaultValue": 1,
  "description": "Controls logging verbosity.",
  "name": "verbose",
  "type": "Int"
 }
}


def valmod(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="VALMOD",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/valmod",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_valmod,
        param_schema=_valmod_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
