from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter


_sr_parameters = {
 "mag_window_size": {
  "defaultValue": 3,
  "description": "Window size for sliding window average calculation",
  "name": "mag_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "score_window_size": {
  "defaultValue": 40,
  "description": "Window size for anomaly scoring",
  "name": "score_window_size",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 50,
  "description": "Sliding window size",
  "name": "window_size",
  "type": "int"
 }
}


def sr(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="SR",
        main=DockerAdapter(
            image_name="mut:5000/akita/sr",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_sr_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
