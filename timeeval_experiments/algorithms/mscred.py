from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality
from .common import SKIP_PULL, DEFAULT_TIMEOUT

import numpy as np


import numpy as np
from timeeval.utils.window import ReverseWindowing
# post-processing for MSCRED
def post_mscred(scores: np.ndarray, args: dict) -> np.ndarray:
    ds_length = np.genfromtxt(args.get("dataInput", "")).shape[0]-1  # subtract header line
    gap_time = args.get("hyper_params", {}).get("gap_time", 10)
    window_size = args.get("hyper_params", {}).get("window_size", 5)
    max_window_size = max(args.get("hyper_params", {}).get("windows", [10, 30, 60]))
    offset = (ds_length - (max_window_size - 1)) % gap_time
    image_scores = ReverseWindowing(window_size=window_size).fit_transform(scores)
    return np.concatenate([np.repeat(image_scores[:-offset], gap_time), image_scores[-offset:]])


def mscred(params: Any = None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="MSCRED-docker",
        main=DockerAdapter(
            image_name="mut:5000/akita/mscred",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_mscred,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
