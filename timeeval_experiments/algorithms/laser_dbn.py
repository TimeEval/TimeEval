from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter


_laser_dbn_parameters = {
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


def laser_dbn(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="LaserDBN",
        main=DockerAdapter(
            image_name="mut:5000/akita/laser_dbn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_laser_dbn_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
