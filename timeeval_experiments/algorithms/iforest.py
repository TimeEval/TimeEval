from durations import Duration
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def iforest(params=None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="iForest-docker",
        main=DockerAdapter(image_name="mut:5000/akita/iforest", skip_pull=skip_pull, timeout=timeout),
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
