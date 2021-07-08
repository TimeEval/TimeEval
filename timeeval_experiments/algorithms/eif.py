from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter


_eif_parameters = {
 "extension_level": {
  "defaultValue": None,
  "description": "Extension level 0 resembles standard isolation forest. If unspecified (`None`), then `extension_level=X.shape[1] - 1`.",
  "name": "extension_level",
  "type": "int"
 },
 "limit": {
  "defaultValue": None,
  "description": "The maximum allowed tree depth. This is by default set to average length of unsucessful search in a binary tree.",
  "name": "limit",
  "type": "int"
 },
 "max_samples": {
  "defaultValue": None,
  "description": "The number of samples to draw from X to train each base estimator: `max_samples * X.shape[0]`. If unspecified (`None`), then `max_samples=min(256, X.shape[0])`.",
  "name": "max_samples",
  "type": "float"
 },
 "n_trees": {
  "defaultValue": 200,
  "description": "The number of decision trees (base estimators) in the forest (ensemble).",
  "name": "n_trees",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def eif(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Extended Isolation Forest",
        main=DockerAdapter(
            image_name="mut:5000/akita/eif",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_eif_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
