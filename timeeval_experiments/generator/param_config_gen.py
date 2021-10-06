import json
import os
import warnings
from pathlib import Path
from typing import Union, Dict, Any

from .parameter_matrix_parsing import ParameterMatrixProxy


class ParamConfigGenerator:
    FIXED_KEY = "fixed_params"
    SHARED_KEY = "shared_params"
    DEPENDENT_KEY = "dependent_params"
    OPTIMIZED_KEY = "optimized_params"
    HEURISTIC_MAPPING_KEY = "__heuristic_function_mapping"
    OVERWRITES_KEY = "__algorithm_overwrites"

    def __init__(self, matrix_path: Union[str, Path]):
        self.pmp = ParameterMatrixProxy(matrix_path)

    def generate_template(self, target: Union[str, Path]) -> None:
        target = Path(target)
        config = {
            self.FIXED_KEY: self.pmp.fixed_params(),
            self.SHARED_KEY: self.pmp.shared_params(),
            self.DEPENDENT_KEY: self.pmp.dependent_params(),
            self.OPTIMIZED_KEY: self.pmp.optimized_params(),
            self.HEURISTIC_MAPPING_KEY: {},
            self.OVERWRITES_KEY: {}
        }
        self._write(config, target)

    def generate(self, target: Union[str, Path], overwrite: bool = False) -> None:
        target = Path(target)
        if overwrite or not target.exists():
            self.generate_template(target)
            return

        config = {}
        if target.exists() and target.is_file():
            with target.open("r") as fh:
                config = json.load(fh)

        config[self.FIXED_KEY] = self.pmp.fixed_params()
        if self.SHARED_KEY in config:
            self._merge_shared(config)
        else:
            config[self.SHARED_KEY] = self.pmp.shared_params()
        config[self.DEPENDENT_KEY] = self.pmp.dependent_params()
        if self.OPTIMIZED_KEY in config:
            self._merge_optimized(config)
        else:
            config[self.OPTIMIZED_KEY] = self.pmp.optimized_params()

        self._write(config, target)

    def _merge_shared(self, config: Dict[str, Any]) -> None:
        shared_params = config[self.SHARED_KEY]
        new_shared_params = self.pmp.shared_params()
        params = set(list(shared_params.keys()) + list(new_shared_params.keys()))
        for param in params:
            if param in shared_params and param in new_shared_params:
                shared_params[param]["algorithms"] = new_shared_params[param]["algorithms"]
                shared_params[param]["search_space"] = new_shared_params[param]["search_space"]
            elif param not in shared_params:
                shared_params[param] = new_shared_params[param]
            else:  # param not in new_shared_params:
                del shared_params[param]
        config[self.SHARED_KEY] = shared_params

    def _merge_optimized(self, config: Dict[str, Any]) -> None:
        optim_params = config[self.OPTIMIZED_KEY]
        new_optim_params = self.pmp.optimized_params()
        params = set(list(optim_params.keys()) + list(new_optim_params.keys()))
        for param in params:
            if param not in new_optim_params:
                del optim_params[param]
                continue

            if param in new_optim_params:
                new_param_config = new_optim_params[param]
                if isinstance(new_param_config, dict) and "MANUAL" in new_param_config.values():
                    if param in optim_params and isinstance(optim_params[param], dict):
                        warnings.warn(f"{self.OPTIMIZED_KEY}: Found 'MANUAL' marker for parameter {param}. "
                                      "Using existing value(s).")
                        param_config = optim_params[param]
                        to_change_algos = []
                        for algo in new_param_config:
                            if new_param_config[algo] == "MANUAL" and algo not in param_config:
                                to_change_algos.append(algo)
                        for algo in to_change_algos:
                            param_config[algo] = new_param_config[algo]
                            warnings.warn(f"{self.OPTIMIZED_KEY}: Found 'MANUAL' marker for parameter {param} and "
                                          f"algorithm {algo}. Please set value(s) after the generation step manually!")
                        continue
                    else:
                        warnings.warn(f"{self.OPTIMIZED_KEY}: Found 'MANUAL' marker for parameter {param}. Please "
                                      "set value(s) after the generation step manually!")

            # for everything else:
            optim_params[param] = new_optim_params[param]

        config[self.OPTIMIZED_KEY] = optim_params

    @staticmethod
    def _write(config: Dict[str, Any], target: Path) -> None:
        with target.open("w") as fh:
            json.dump(config, fh, sort_keys=True, indent=2)
            fh.write(os.linesep)


if __name__ == "__main__":
    p = ParamConfigGenerator("timeeval_experiments/parameter-matrix.csv")
    p.generate("timeeval_experiments/params.json")
