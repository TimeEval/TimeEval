import numpy as np
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.utils.window import ReverseWindowing
from .common import SKIP_PULL


def _post_s2g(scores: np.ndarray, args: dict) -> np.ndarray:
    params = args.get("hyper_params", {})
    window_size = params.get("window_size", 50)
    query_window_size = params.get("query_window_size", 75)
    convolution_size = params.get("convolution_size", window_size // 3)
    size = (window_size + convolution_size) + query_window_size + 4
    return ReverseWindowing(window_size=size).fit_transform(scores)


def series2graph(params=None, skip_pull: bool = SKIP_PULL) -> Algorithm:
    return Algorithm(
        name="S2G-docker",
        main=DockerAdapter(image_name="mut:5000/akita/series2graph", skip_pull=skip_pull),
        postprocess=_post_s2g,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
