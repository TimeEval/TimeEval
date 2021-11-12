from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig, FullParameterGrid


_baseline_normal_parameters: Dict[str, Dict[str, Any]] = {}


def baseline_normal(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Normal Baseline",
        main=DockerAdapter(
            image_name="mut:5000/akita/baseline_normal",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        params=_baseline_normal_parameters,
        param_grid=params or FullParameterGrid({}),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
