import numpy as np
from durations import Duration
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm, AlgorithmParameter
from timeeval.adapters import DockerAdapter
from timeeval.utils.window import ReverseWindowing
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def _post_torsk(data: AlgorithmParameter, args: dict) -> np.ndarray:
    if isinstance(data, np.ndarray):
        scores = data
    else:
        scores = np.genfromtxt(data)
    pred_size = args.get("hyper_params", {}).get("pred_size", 20)
    context_window_size = args.get("hyper_params", {}).get("context_window_size", 10)
    size = pred_size * context_window_size + 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


def torsk(params=None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="Torsk-docker",
        main=DockerAdapter(image_name="mut:5000/akita/torsk", skip_pull=skip_pull, timeout=timeout),
        postprocess=_post_torsk,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
