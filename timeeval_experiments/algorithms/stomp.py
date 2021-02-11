import numpy as np
from durations import Duration
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.utils.window import ReverseWindowing
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def _post_stomp(scores: np.ndarray, args: dict) -> np.ndarray:
    params = args.get("hyper_params", {})
    window_size = params.get("window_size", 30)
    scores_new = ReverseWindowing(window_size=window_size).fit_transform(scores)
    return scores_new


def stomp(params=None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="STOMP-docker",
        main=DockerAdapter(image_name="mut:5000/akita/stomp", skip_pull=skip_pull, timeout=timeout),
        postprocess=_post_stomp,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
