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


def pci(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="PCI",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/pci",
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
