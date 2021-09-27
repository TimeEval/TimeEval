from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for stomp
def post_stomp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_stomp_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 30,
  "description": "Size of the sliding window.",
  "name": "anomaly_window_size",
  "type": "Int"
 },
 "exclusion_zone": {
  "defaultValue": 0.5,
  "description": "Size of the exclusion zone as a factor of the window_size. This prevents self-matches.",
  "name": "exclusion_zone",
  "type": "Float"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of jobs to run in parallel. `-1` is not supported, defaults back to serial implementation.",
  "name": "n_jobs",
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


def stomp(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="STOMP",
        main=DockerAdapter(
            image_name="mut:5000/akita/stomp",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_stomp,
        params=_stomp_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )