import numpy as np
from durations import Duration
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.utils.window import ReverseWindowing
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def _post_valmod(scores: np.ndarray, args: dict) -> np.ndarray:
    window_length = args.get("hyper_params", {}).get("window_min", 30)
    return ReverseWindowing(window_size=window_length).fit_transform(scores)


def valmod(params=None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="VALMOD-docker",
        main=DockerAdapter(image_name="mut:5000/akita/valmod", skip_pull=skip_pull, timeout=timeout),
        postprocess=_post_valmod,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
