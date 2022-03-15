from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for s2g
def post_s2g(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 50)
    query_window_size = args.get("hyper_params", {}).get("query_window_size", 75)
    convolution_size = args.get("hyper_params", {}).get("convolution_size", window_size // 3)
    size = (window_size + convolution_size) + query_window_size + 4
    return ReverseWindowing(window_size=size).fit_transform(scores)


_series2graph_parameters: Dict[str, Dict[str, Any]] = {
 "query_window_size": {
  "defaultValue": 75,
  "description": "Size of the sliding windows used to find anomalies (query subsequences). query_window_size must be >= window_size! (paper: `l_q`)",
  "name": "query_window_size",
  "type": "Int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "Int"
 },
 "rate": {
  "defaultValue": 30,
  "description": "Number of angles used to extract pattern nodes. A higher value will lead to high precision, but at the cost of increased computation time. (paper: `r` performance parameter)",
  "name": "rate",
  "type": "Int"
 },
 "window_size": {
  "defaultValue": 50,
  "description": "Size of the sliding window (paper: `l`), independent of anomaly length, but should in the best case be larger.",
  "name": "window_size",
  "type": "Int"
 }
}


def series2graph(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Series2Graph",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/series2graph",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_s2g,
        param_schema=_series2graph_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
