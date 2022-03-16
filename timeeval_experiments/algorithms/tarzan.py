from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for TARZAN
def post_tarzan(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_tarzan_parameters: Dict[str, Dict[str, Any]] = {
 "alphabet_size": {
  "defaultValue": 4,
  "description": "Number of symbols used for discretization by SAX (performance parameter)",
  "name": "alphabet_size",
  "type": "int"
 },
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "Size of the sliding window. Equal to the discord length!",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def tarzan(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="TARZAN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/tarzan",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_tarzan,
        param_schema=_tarzan_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
