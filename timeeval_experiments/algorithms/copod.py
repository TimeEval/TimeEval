from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter


_copod_parameters: Dict[str, Dict[str, Any]] = {
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def copod(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="COPOD",
        main=DockerAdapter(
            image_name="mut:5000/akita/copod",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_copod_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
