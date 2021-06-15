from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.data_types import TrainingType, InputDimensionality
from .common import SKIP_PULL, DEFAULT_TIMEOUT

import numpy as np


import numpy as np

def post_img_embedding_cae(scores: np.ndarray, args: dict) -> np.ndarray:
    dataset_len = np.genfromtxt(args["inputData"], skip_header=1, delimiter=",", usecols=[1]).shape[0]
    window_size = args.get("customParameters", {}).get("window_size", 128)
    return np.repeat(scores, window_size)[:dataset_len]


def img_embedding_cae(params: Any = None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="ImageEmbeddingCAE-docker",
        main=DockerAdapter(
            image_name="mut:5000/akita/img_embedding_cae",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_img_embedding_cae,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
