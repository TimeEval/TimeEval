from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import FullParameterGrid


_robust_pca_parameters: Dict[str, Dict[str, Any]] = {
 "max_iter": {
  "defaultValue": 1000,
  "description": "Defines the number of maximum robust PCA iterations for solving matrix decomposition.",
  "name": "max_iter",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def robust_pca(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="RobustPCA",
        main=DockerAdapter(
            image_name="mut:5000/akita/robust_pca",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_robust_pca_parameters,
        param_grid=FullParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
