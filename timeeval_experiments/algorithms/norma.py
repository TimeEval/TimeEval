import numpy as np
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.utils.window import ReverseWindowing
from .common import SKIP_PULL


def _post_norma(scores: np.ndarray, args: dict) -> np.ndarray:
    params = args.get("hyper_params", {})
    size = 2 * (params.get("window_size", 20) - 1) + 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


def norma(params=None, skip_pull: bool = SKIP_PULL) -> Algorithm:
    return Algorithm(
        name="NormA-docker",
        main=DockerAdapter(image_name="mut:5000/akita/norma", skip_pull=skip_pull),
        postprocess=_post_norma,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
