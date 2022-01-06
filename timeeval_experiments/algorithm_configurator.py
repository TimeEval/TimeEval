import json
import warnings
from pathlib import Path
from typing import Union, TypeVar, List, Dict, Any

from timeeval import Algorithm
from timeeval.heuristics import TimeEvalHeuristic
from timeeval.params import FullParameterGrid, IndependentParameterGrid
from timeeval_experiments.generator import ParamConfigGenerator


T = TypeVar("T")


class AlgorithmConfigurator:
    def __init__(self, config_path: Union[str, Path], check: bool = True):
        self.config_path = Path(config_path)
        with open(config_path, "r") as fh:
            config = json.load(fh)
        self._fixed_params = config[ParamConfigGenerator.FIXED_KEY]
        self._dependent_params = config[ParamConfigGenerator.DEPENDENT_KEY]
        self._shared_params = config[ParamConfigGenerator.SHARED_KEY]
        self._optimized_params = config[ParamConfigGenerator.OPTIMIZED_KEY]
        self._algorithm_overwrites = config[ParamConfigGenerator.OVERWRITES_KEY]
        self._heuristic_mapping: Dict[str, str] = config[ParamConfigGenerator.HEURISTIC_MAPPING_KEY]
        if check and self._heuristic_mapping:
            self._check_heuristics()

    def _check_heuristics(self):
        broken_heuristics = []
        for key in self._heuristic_mapping:
            signature = self._heuristic_mapping[key]
            try:
                TimeEvalHeuristic(signature)
            except Exception as e:
                broken_heuristics.append((key, e))
        if len(broken_heuristics) > 0:
            for k, ex in broken_heuristics:
                print(f"Heuristic '{k}' is invalid: {ex}")
            raise SyntaxError("Heuristics mapping contains invalid entries, please consider log output!")
        else:
            print("Heuristics are valid.")

    def _substitute_heuristics(self, search_space: List[Any], checked: bool = False) -> Any:
        def substitute(v):
            try:
                return f"heuristic:{self._heuristic_mapping[v]}"
            except (KeyError, TypeError) as e:
                if checked:
                    raise ValueError(f"Entry {v} is not a valid heuristic!") from e
                else:
                    return v
        return [substitute(value) for value in search_space]

    @staticmethod
    def wrap(elems: Union[List[T], T]) -> List[T]:
        if not isinstance(elems, list):
            return [elems]
        else:
            return elems

    def configure(self, algos: Union[List[Algorithm], Algorithm],
                  use_defaults: bool = False,
                  ignore_overwrites: bool = False,
                  ignore_fixed: bool = False,
                  ignore_dependent: bool = False,
                  ignore_shared: bool = False,
                  ignore_optimized: bool = False,
                  perform_search: bool = True,
                  assume_parameter_independence: bool = False) -> None:
        if use_defaults:
            return

        algos = self.wrap(algos)
        for algo in algos:
            configured_params = {}
            prio_params = {}
            if algo.name in self._algorithm_overwrites:
                prio_params = self._algorithm_overwrites[algo.name]

            for p in algo.param_schema:
                if not ignore_overwrites and p in prio_params:
                    value = prio_params[p]
                    if value != "default":
                        # don't allow specifying a search space; assume lists are parameters of type list
                        value = [value]

                        # map heuristics
                        value = self._substitute_heuristics(value)
                        configured_params[p] = value
                    #  else: don't specify a value, because the default is used anyway

                elif not ignore_fixed and p in self._fixed_params:
                    value = self._fixed_params[p]
                    if value != "default":
                        configured_params[p] = [value]
                    #  else: don't specify a value, because the default is used anyway

                elif not ignore_shared and p in self._shared_params:
                    value = self._shared_params[p]["value"]
                    # this should be a single fixed value
                    if isinstance(value, list):
                        raise ValueError(f"Wrong format: value for shared parameter '{p}' "
                                         "should be a single parameter value")
                    configured_params[p] = [value]

                elif perform_search and p in self._shared_params:
                    value = self._shared_params[p]["search_space"]
                    # this should already be a list of parameter options
                    if not isinstance(value, list):
                        raise ValueError(f"Wrong format: search_space for shared parameter '{p}' "
                                         "should be a list of parameter options")
                    configured_params[p] = value

                elif perform_search and not ignore_optimized and p in self._optimized_params:
                    value = self._optimized_params[p]
                    # if there are multiple algos with the same parameter, there can be different search spaces
                    if isinstance(value, dict):
                        value = value[algo.name]
                    # this should already be a list of parameter options
                    if not isinstance(value, list):
                        raise ValueError(f"Wrong format: value for optimized parameter '{p}' ({algo.name}) "
                                         "should be a list of parameter options")

                    configured_params[p] = self._substitute_heuristics(value)

                elif not ignore_dependent and p in self._dependent_params:
                    value = self._dependent_params[p]
                    value = self.wrap(value)
                    configured_params[p] = self._substitute_heuristics(value, checked=True)

                else:
                    warnings.warn(f"{algo.name}: Cannot configure parameter {p}, because no configuration value was found! "
                                  "Using default.")

            if assume_parameter_independence:
                # assume that fixed parameters are defaults and should be set for all configurations
                default_params = {}
                search_params = {}
                for p in configured_params:
                    value = configured_params[p]
                    if isinstance(value, list) and len(value) > 1:
                        search_params[p] = value
                    else:
                        value = value[0]
                        # convert list default params to tuples
                        if isinstance(value, list):
                            value = tuple(value)
                        default_params[p] = value
                algo.param_config = IndependentParameterGrid(search_params, default_params=default_params)
            else:
                algo.param_config = FullParameterGrid(configured_params)
