from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig, FullParameterGrid


_bagel_parameters: Dict[str, Dict[str, Any]] = {
 "cuda": {
  "defaultValue": "False",
  "description": "Use GPU for training",
  "name": "cuda",
  "type": "boolean"
 },
 "early_stopping_delta": {
  "defaultValue": 0.05,
  "description": "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop",
  "name": "early_stopping_delta",
  "type": "float"
 },
 "early_stopping_patience": {
  "defaultValue": 10,
  "description": "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop",
  "name": "early_stopping_patience",
  "type": "int"
 },
 "epochs": {
  "defaultValue": 50,
  "description": "Number of passes over the entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "latent_size": {
  "defaultValue": 8,
  "description": "Dimensionality of encoding",
  "name": "latent_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Percentage of training data to use for training",
  "name": "split",
  "type": "float"
 }
}


def bagel(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Bagel",
        main=DockerAdapter(
            image_name="mut:5000/akita/bagel",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_bagel_parameters,
        param_grid=params or FullParameterGrid({}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
