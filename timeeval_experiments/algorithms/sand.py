from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_sand_parameters: Dict[str, Dict[str, Any]] = {
 "alpha": {
  "defaultValue": 0.5,
  "description": "Weight decay / forgetting factor. Quite robust",
  "name": "alpha",
  "type": "float"
 },
 "anomaly_window_size": {
  "defaultValue": 75,
  "description": "Size of the anomalous pattern; sliding windows for clustering and preprocessing are of size 3*anomaly_window_size.",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "iter_batch_size": {
  "defaultValue": 500,
  "description": "Number of points for each batch. Mostly impacts performance (not too small).",
  "name": "iter_batch_size",
  "type": "int"
 },
 "n_clusters": {
  "defaultValue": 6,
  "description": "Number of clusters used in Kshape that are maintained iteratively as a normal model",
  "name": "n_clusters",
  "type": "int"
 },
 "n_init_train": {
  "defaultValue": 2000,
  "description": "Number of points to build the initial model (may contain anomalies)",
  "name": "n_init_train",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def sand(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="SAND",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/sand",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_sand_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
