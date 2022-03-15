from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for norma
def _post_norma(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    size = 2 * window_size - 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


_norma_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "Sliding window size used to create subsequences (equal to desired anomaly length)",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "normal_model_percentage": {
  "defaultValue": 0.5,
  "description": "Percentage of (random) subsequences used to build the normal model.",
  "name": "normal_model_percentage",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def norma(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="NormA",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/norma",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=_post_norma,
        param_schema=_norma_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
