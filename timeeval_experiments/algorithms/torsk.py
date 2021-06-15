from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality
from .common import SKIP_PULL, DEFAULT_TIMEOUT

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for Torsk
def _post_torsk(scores: np.ndarray, args: dict) -> np.ndarray:
    pred_size = args.get("hyper_params", {}).get("pred_size", 20)
    context_window_size = args.get("hyper_params", {}).get("context_window_size", 10)
    size = pred_size * context_window_size + 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


def torsk(params: Any = None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="Torsk-docker",
        main=DockerAdapter(
            image_name="mut:5000/akita/torsk",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=_post_torsk,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
