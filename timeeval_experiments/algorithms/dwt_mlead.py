from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_dwt_mlead_parameters: Dict[str, Dict[str, Any]] = {
 "quantile_epsilon": {
  "defaultValue": 0.01,
  "description": "Percentage of windows to flag as anomalous within each decomposition level's coefficients",
  "name": "quantile_epsilon",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "start_level": {
  "defaultValue": 3,
  "description": "First discrete wavelet decomposition level to consider",
  "name": "start_level",
  "type": "int"
 }
}


def dwt_mlead(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DWT-MLEAD",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/dwt_mlead",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_dwt_mlead_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
