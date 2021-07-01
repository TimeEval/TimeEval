from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for valmod
def post_valmod(scores: np.ndarray, args: dict) -> np.ndarray:
    window_length = args.get("hyper_params", {}).get("window_min", 30)
    return ReverseWindowing(window_size=window_length).fit_transform(scores)


_valmod_parameters = {
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


def valmod(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="VALMOD",
        main=DockerAdapter(
            image_name="mut:5000/akita/valmod",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_valmod,
        params=_valmod_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
