from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TimeEvalParameterHeuristic


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from typing import Tuple
    from pathlib import Path
    from ..algorithm import Algorithm
    from ..datasets import Dataset


class DatasetIdHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to pass the dataset ID as a parameter value.

    The dataset ID is a tuple of the collection name and the dataset name, such as
    ``("KDD-TSAD", "022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4")``.

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
    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Tuple[str, str]:  # type: ignore[no-untyped-def]
        return dataset_details.datasetId
