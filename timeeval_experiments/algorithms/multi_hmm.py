from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality


_multi_hmm_parameters = {
 "discretizer": {
  "defaultValue": "fcm",
  "description": "Available discretizers are \"sugeno\", \"choquet\", and \"fcm\". If only 1 feature in time series, K-Bins discretizer is used.",
  "name": "discretizer",
  "type": "enum[sugeno,choquet,fcm]"
 },
 "n_bins": {
  "defaultValue": 10,
  "description": "Number of bins used for discretization.",
  "name": "n_bins",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def multi_hmm(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="MultiHMM",
        main=DockerAdapter(
            image_name="mut:5000/akita/multi_hmm",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_multi_hmm_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
