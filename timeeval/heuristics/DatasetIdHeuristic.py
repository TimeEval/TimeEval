from pathlib import Path
from typing import Tuple

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class DatasetIdHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to pass the dataset ID as a parameter value.

    Examples
    --------
    >>> from timeeval import Algorithm, TrainingType, InputDimensionality
    >>> from timeeval.adapters import FunctionAdapter
    >>> from timeeval.params import FixedParameters
    >>> algo = Algorithm(name="example", main=FunctionAdapter(lambda x, args: x),
    ...     training_type=TrainingType.UNSUPERVISED,
    ...     input_dimensionality=InputDimensionality.UNIVARIATE,
    ...     param_config=FixedParameters({"dataset_id": "heuristic:DatasetIdHeuristic()"},
    ... )
    """
    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Tuple[str, str]:
        return dataset_details.datasetId
