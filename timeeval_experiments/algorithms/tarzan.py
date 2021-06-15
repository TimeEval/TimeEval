from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def tarzan(params: Any = None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="TARZAN-docker",
        main=DockerAdapter(
            image_name="mut:5000/akita/tarzan",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
