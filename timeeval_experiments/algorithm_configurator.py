import json
import warnings
from pathlib import Path
from typing import Union, TypeVar, List

from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm
from timeeval_experiments.generator.param_config_gen import ParamConfigGenerator


T = TypeVar("T")


class AlgorithmConfigurator:
    def __init__(self, config_path: Union[str, Path]):
        config_path = Path(config_path)
        with open(config_path, "r") as fh:
            config = json.load(fh)
            self._fixed_params = config[ParamConfigGenerator.FIXED_KEY]
            self._dependent_params = config[ParamConfigGenerator.DEPENDENT_KEY]
            self._shared_params = config[ParamConfigGenerator.SHARED_KEY]
            self._optimized_params = config[ParamConfigGenerator.OPTIMIZED_KEY]
            self._algorithm_overwrites = config[ParamConfigGenerator.OVERWRITES_KEY]

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
                  perform_search: bool = True) -> None:
        if use_defaults:
            return

        algos = self.wrap(algos)
        for algo in algos:
            configured_params = {}
            prio_params = {}
            if algo.name in self._algorithm_overwrites:
                prio_params = self._algorithm_overwrites[algo.name]

            for p in algo.params:
                if not ignore_overwrites and p in prio_params:
                    value = prio_params[p]
                    # allow specifying a search space or a fixed value
                    if not isinstance(value, list):
                        value = [value]
                    configured_params[p] = value
                elif not ignore_fixed and p in self._fixed_params:
                    value = self._fixed_params[p]
                    if value != "default":
                        configured_params[p] = [value]
                    #  else: don't specify a value, because the default is used anyway
                elif perform_search and p in self._shared_params:
                    # this should already be a list of parameter options
                    configured_params[p] = self._shared_params[p]["search_space"]
                elif perform_search and p in self._optimized_params:
                    value = self._optimized_params[p]
                    # if there are multiple algos with the same parameter, there can be different search spaces
                    if isinstance(value, dict):
                        value = value[algo.name]
                    # this should already be a list of parameter options
                    configured_params[p] = value
                elif not ignore_dependent and p in self._dependent_params:
                    warnings.warn(f"There are no heuristics for dependent parameters yet (param={p})! Using default.")
                else:
                    warnings.warn(f"Cannot configure parameter {p}, because no configuration value was found! Using default.")

            algo.param_grid = ParameterGrid(configured_params)
