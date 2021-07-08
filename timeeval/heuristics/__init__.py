import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .AnomalyLengthHeuristic import AnomalyLengthHeuristic
from .CleanStartSequenceSizeHeuristic import CleanStartSequenceSizeHeuristic
from .EmbedDimRangeHeuristic import EmbedDimRangeHeuristic
from .ParameterDependenceHeuristic import ParameterDependenceHeuristic
from .PeriodSizeHeuristic import PeriodSizeHeuristic
from .RelativeDatasetSizeHeuristic import RelativeDatasetSizeHeuristic
from .base import TimeEvalParameterHeuristic


def _check_signature(signature: str) -> bool:
    res = re.fullmatch(
        r"^(RelativeDatasetSizeHeuristic|AnomalyLengthHeuristic|CleanStartSequenceSizeHeuristic|ParameterDependenceHeuristic|PeriodSizeHeuristic|EmbedDimRangeHeuristic)[(].*[)]$",
        signature,
        re.M
    )
    return res is not None


def TimeEvalHeuristic(signature: str) -> TimeEvalParameterHeuristic:
    """This wrapper allows using the heuristics by name without the need for imports."""
    if not _check_signature(signature):
        raise ValueError(f"Heuristic '{signature}' is invalid! Only constructor calls to classes derived from "
                         "TimeEvalParameterHeuristic are allowed.")
    return eval(signature)


def inject_heuristic_values(
        params: Dict[str, Any],
        algorithm: Algorithm,
        dataset_details: Dataset,
        dataset_path: Path,
) -> Dict[str, Any]:
    updated_params = deepcopy(params)
    # defer dependence heuristics after all other heuristics
    heuristic_params = {(k, v) for k, v in params.items() if isinstance(v, str) and v.startswith("heuristic:")}
    deferred_params = {(k, v) for k, v in heuristic_params if "ParameterDependenceHeuristic" in v}
    heuristic_params -= deferred_params
    for k, v in list(heuristic_params) + list(deferred_params):
        if isinstance(v, str) and v.startswith("heuristic:"):
            heuristic_signature: str = ":".join(v.split(":")[1:]).strip()
            new_value = TimeEvalHeuristic(heuristic_signature)(algorithm,
                                                               dataset_details,
                                                               dataset_path,
                                                               params=updated_params)
            if new_value is None:
                del updated_params[k]
            else:
                updated_params[k] = new_value
    return updated_params
