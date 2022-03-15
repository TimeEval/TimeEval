from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_laser_dbn_parameters: Dict[str, Dict[str, Any]] = {
 "n_bins": {
  "defaultValue": 10,
  "description": "Number of bins used for discretization.",
  "name": "n_bins",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "timesteps": {
  "defaultValue": 2,
  "description": "Number of time steps the DBN builds probabilities for (min: 2)",
  "name": "timesteps",
  "type": "int"
 }
}


def laser_dbn(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="LaserDBN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/laser_dbn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_laser_dbn_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
