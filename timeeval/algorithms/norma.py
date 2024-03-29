# DO NOT EDIT THIS FILE!
# This file was automatically generated using the timeeval_experiments.generator from the template:
# timeeval_experiments/generator/templates/docker-algorithm.py.jinja
from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for norma
def _post_norma(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    size = 2 * window_size - 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


_norma_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "Sliding window size used to create subsequences (equal to desired anomaly length)",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "normal_model_percentage": {
  "defaultValue": 0.5,
  "description": "Percentage of (random) subsequences used to build the normal model.",
  "name": "normal_model_percentage",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def norma(params: Optional[ParameterConfig] = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    """NormA

    Improved algorithm based on NorM (https://doi.org/10.1109/ICDE48307.2020.00182).

    .. warning::
       The implementation of this algorithm is not publicly available (closed source).
       Thus, TimeEval will fail to download the Docker image and the algorithm will not be available.
       Please contact the authors of the algorithm for the implementation and build the algorithm Docker image yourself.

    **Algorithm Parameters:**

    anomaly_window_size: int
        Sliding window size used to create subsequences (equal to desired anomaly length) (default: ``20``)
    normal_model_percentage: float
        Percentage of (random) subsequences used to build the normal model. (default: ``0.5``)
    random_state: int
        Seed for random number generation. (default: ``42``)

    Parameters
    ----------
    params : Optional[ParameterConfig]
        Parameter configuration for the algorithm
    skip_pull : bool
        Set to ``True`` to skip pulling the Docker image and use a local image instead.
        If the image is not present locally, this will raise an error.
    timeout : Optional[Duration]
        Set an individual execution and training timeout for this algorithm.
        This will overwrite the global timeouts set using :class:`~timeeval.ResourceConstraints`.

    Returns
    -------
    ~timeeval.Algorithm
        A correctly configured :class:`~timeeval.Algorithm` object for the NormA algorithm.
    """
    return Algorithm(
        name="NormA",
        main=DockerAdapter(
            image_name="ghcr.io/timeeval/norma",
            tag="0.3.0",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=_post_norma,
        param_schema=_norma_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
