# DO NOT EDIT THIS FILE!
# This file was automatically generated using the timeeval_experiments.generator from the template:
# timeeval_experiments/generator/templates/docker-algorithm.py.jinja
from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_pci_parameters: Dict[str, Dict[str, Any]] = {
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "thresholding_p": {
  "defaultValue": 0.05,
  "description": "This parameter is only needed if the algorithm should decide itself whether a point is an anomaly. It treats `p` as a confidence coefficient. It's the t-statistics confidence coefficient. The smaller `p` is, the bigger is the confidence interval. If `p` is too small, anomalies might not be found. If `p` is too big, too many points might be labeled anomalous.",
  "name": "thresholding_p",
  "type": "float"
 },
 "window_size": {
  "defaultValue": 20,
  "description": "The algorithm uses windows around the current points to predict that point (`k` points before and `k` after, where `k = window_size // 2`). The difference between real and predicted value is used as anomaly score. The parameter `window_size` acts as a kind of smoothing factor. The bigger the `window_size`, the smoother the predictions, the more values have big errors. If `window_size` is too small, anomalies might not be found. `window_size` should correlate with anomaly window sizes.",
  "name": "window_size",
  "type": "int"
 }
}


def pci(params: Optional[ParameterConfig] = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    """PCI

    Implementation of https://doi.org/10.1155/2014/879736


    **Algorithm Parameters:**

    window_size: int
        The algorithm uses windows around the current points to predict that point (`k` points before and `k` after, where `k = window_size // 2`). The difference between real and predicted value is used as anomaly score. The parameter `window_size` acts as a kind of smoothing factor. The bigger the `window_size`, the smoother the predictions, the more values have big errors. If `window_size` is too small, anomalies might not be found. `window_size` should correlate with anomaly window sizes. (default: ``20``)
    thresholding_p: float
        This parameter is only needed if the algorithm should decide itself whether a point is an anomaly. It treats `p` as a confidence coefficient. It's the t-statistics confidence coefficient. The smaller `p` is, the bigger is the confidence interval. If `p` is too small, anomalies might not be found. If `p` is too big, too many points might be labeled anomalous. (default: ``0.05``)
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
        A correctly configured :class:`~timeeval.Algorithm` object for the PCI algorithm.
    """
    return Algorithm(
        name="PCI",
        main=DockerAdapter(
            image_name="ghcr.io/timeeval/pci",
            tag="0.3.0",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_pci_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
