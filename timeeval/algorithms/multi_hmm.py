# DO NOT EDIT THIS FILE!
# This file was automatically generated using the timeeval_experiments.generator from the template:
# timeeval_experiments/generator/templates/docker-algorithm.py.jinja
from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_multi_hmm_parameters: Dict[str, Dict[str, Any]] = {
 "discretizer": {
  "defaultValue": "fcm",
  "description": "Available discretizers are \"sugeno\", \"choquet\", and \"fcm\". If only 1 feature in time series, K-Bins discretizer is used.",
  "name": "discretizer",
  "type": "enum[sugeno,choquet,fcm]"
 },
 "n_bins": {
  "defaultValue": 10,
  "description": "Number of bins used for discretization.",
  "name": "n_bins",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def multi_hmm(params: Optional[ParameterConfig] = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    """MultiHMM

    Implementation of https://doi.org/10.1016/j.asoc.2017.06.035


    **Algorithm Parameters:**

    discretizer: enum[sugeno,choquet,fcm]
        Available discretizers are "sugeno", "choquet", and "fcm". If only 1 feature in time series, K-Bins discretizer is used. (default: ``fcm``)
    n_bins: int
        Number of bins used for discretization. (default: ``10``)
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
        A correctly configured :class:`~timeeval.Algorithm` object for the MultiHMM algorithm.
    """
    return Algorithm(
        name="MultiHMM",
        main=DockerAdapter(
            image_name="ghcr.io/timeeval/multi_hmm",
            tag="0.3.0",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_multi_hmm_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
