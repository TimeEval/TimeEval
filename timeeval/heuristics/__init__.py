from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, MutableMapping

from .base import TimeEvalParameterHeuristic
from .AnomalyLengthHeuristic import AnomalyLengthHeuristic
from .CleanStartSequenceSizeHeuristic import CleanStartSequenceSizeHeuristic
from .ContaminationHeuristic import ContaminationHeuristic
from .DatasetIdHeuristic import DatasetIdHeuristic
from .DefaultExponentialFactorHeuristic import DefaultExponentialFactorHeuristic
from .DefaultFactorHeuristic import DefaultFactorHeuristic
from .EmbedDimRangeHeuristic import EmbedDimRangeHeuristic
from .ParameterDependenceHeuristic import ParameterDependenceHeuristic
from .PeriodSizeHeuristic import PeriodSizeHeuristic
from .RelativeDatasetSizeHeuristic import RelativeDatasetSizeHeuristic


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, TypeVar, Mapping
    from ..algorithm import Algorithm
    from ..datasets import Dataset
    from ..params import Params

    T = TypeVar("T", Params, Mapping[str, Any], MutableMapping[str, Any])


def _check_signature(signature: str) -> bool:
    res = re.fullmatch(
        r"^(RelativeDatasetSizeHeuristic|AnomalyLengthHeuristic|CleanStartSequenceSizeHeuristic|"
        r"ParameterDependenceHeuristic|PeriodSizeHeuristic|EmbedDimRangeHeuristic|ContaminationHeuristic|"
        r"DefaultFactorHeuristic|DefaultExponentialFactorHeuristic|DatasetIdHeuristic)[(].*[)]$",
        signature,
        re.M
    )
    return res is not None


def TimeEvalHeuristic(signature: str) -> TimeEvalParameterHeuristic:
    """This wrapper allows using the heuristics by name without the need for imports."""
    if not _check_signature(signature):
        raise ValueError(f"Heuristic '{signature}' is invalid! Only constructor calls to classes derived from "
                         "TimeEvalParameterHeuristic are allowed.")
    return eval(signature)  # type: ignore


def inject_heuristic_values(
        params: T,
        algorithm: Algorithm,
        dataset_details: Dataset,
        dataset_path: Path,
) -> T:
    # if not hasattr(params, "__setitem__") or not hasattr(params, "__delitem__"):
    if not isinstance(params, MutableMapping):
        # ignore all dynamic parameter search spaces that cannot be altered
        return params

    updated_params = deepcopy(params)
    # defer dependence heuristics after all other heuristics
    heuristic_params = {(k, v) for k, v in params.items() if isinstance(v, str) and v.startswith("heuristic:")}
    deferred_params = {(k, v) for k, v in heuristic_params if "ParameterDependenceHeuristic" in v}
    heuristic_params -= deferred_params
    for k, v in list(heuristic_params) + list(deferred_params):
        if isinstance(v, str) and v.startswith("heuristic:"):
            heuristic_signature: str = ":".join(v.split(":")[1:]).strip()
            heuristic = TimeEvalHeuristic(heuristic_signature)
            try:
                new_value = heuristic(
                    algorithm,
                    dataset_details,
                    dataset_path,
                    # required by ParameterDependenceHeuristic
                    params=updated_params,
                    # required by DefaultFactorHeuristic and DefaultExponentialFactorHeuristic
                    param_name=k
                )
            except Exception as ex:
                raise ValueError(
                    f"Applying heuristic {heuristic_signature} for algorithm {algorithm.name}, target parameter {k}, "
                    f"dataset {dataset_details.datasetId}, and parameters '{updated_params}' failed!"
                ) from ex
            if new_value is None:
                del updated_params[k]
            else:
                updated_params[k] = new_value
    return updated_params
