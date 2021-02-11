import numpy as np
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.utils.window import ReverseWindowing
from .common import SKIP_PULL


def _post_stamp(scores: np.ndarray, args: dict) -> np.ndarray:
    params = args.get("hyper_params", {})
    window_size = params.get("window_size", 30)
    scores_new = ReverseWindowing(window_size=window_size).fit_transform(scores)
    return scores_new


def stamp(params=None, skip_pull: bool = SKIP_PULL) -> Algorithm:
    return Algorithm(
        name="STAMP-docker",
        main=DockerAdapter(image_name="mut:5000/akita/stamp", skip_pull=skip_pull),
        postprocess=_post_stamp,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
