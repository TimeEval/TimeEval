from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from .common import SKIP_PULL


def lof(params=None, skip_pull: bool = SKIP_PULL) -> Algorithm:
    return Algorithm(
        name="LOF-docker",
        main=DockerAdapter(image_name="mut:5000/akita/lof", skip_pull=skip_pull),
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
