import json
from pathlib import Path
from typing import Union, Dict, Any

from timeeval_experiments.generator.parameter_matrix_parsing import ParameterMatrixProxy


def _wip(params: Dict) -> Dict:
    def map(key: str, value: Any):
        if isinstance(value, dict):
            return key, dict([map(k, v) for k, v in value.items()])
        else:
            return key, [value]

    return dict([map(k, v) for k, v in params.items()])


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
            self.OPTIMIZED_KEY: _wip(self.pmp.optimized_params()),
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
        config[self.SHARED_KEY] = self.pmp.shared_params()
        config[self.DEPENDENT_KEY] = self.pmp.dependent_params()
        config[self.OPTIMIZED_KEY] = _wip(self.pmp.optimized_params())

        self._write(config, target)

    @staticmethod
    def _write(config: Dict[str, Any], target: Path) -> None:
        with target.open("w") as fh:
            json.dump(config, fh, sort_keys=True, indent=2)


if __name__ == "__main__":
    p = ParamConfigGenerator("timeeval_experiments/parameter-matrix.csv")
    p.generate("timeeval_experiments/params.json")
