from durations import Duration

from timeeval import Algorithm
from timeeval.adapters import DockerAdapter
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def copod(params=None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    if params:
        import sys
        print("WARN: COPOD does not take parameters!", file=sys.stderr)
    return Algorithm(
        name="COPOD-docker",
        main=DockerAdapter(image_name="mut:5000/akita/copod", skip_pull=skip_pull, timeout=timeout),
        data_as_file=True
    )
