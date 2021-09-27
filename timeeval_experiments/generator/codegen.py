from pathlib import Path
from typing import Union, Any, Dict

import json
from jinja2 import Environment, PackageLoader

from .algorithm_parsing import AlgorithmLoader


class AlgorithmGenerator:
    def __init__(self, timeeval_algorithms: Union[str, Path], skip_pull: bool):
        self.algorithm_details = AlgorithmLoader(timeeval_algorithms)
        self._skip_pull = skip_pull
        self._jenv = Environment(
            loader=PackageLoader("timeeval_experiments", "generator/templates"),
            keep_trailing_newline=True,
            lstrip_blocks=True,
            trim_blocks=True,
        )

    def generate_all(self, target: Union[str, Path], force: bool = False) -> None:
        target_path = Path(target)
        self.generate_init(target_path / "__init__.py", force)
        for algo in self.algorithm_details.algorithm_names:
            self.generate_algorithm(algo, target_path / f"{algo}.py", force)

    def generate_init(self, target: Union[str, Path], force: bool = False) -> None:
        target_path = self._check_target(target, allow_overwrite=force, allow_dir=False)
        file_template = self._jenv.get_template("__init__.py.jinja")
        algorithms = self.algorithm_details.algorithm_names
        with target_path.open("w") as fh:
            fh.write(file_template.render(
                algorithms=algorithms
            ))

    def generate_algorithm(self, algorithm: str, target: Union[str, Path], force: bool = False) -> None:
        target_path = self._check_target(target, allow_overwrite=force, allow_dir=True, name=algorithm)
        file_template = self._jenv.get_template("docker-algorithm.py.jinja")
        algo_data = self.algorithm_details.algo_detail(algorithm)
        if "post_process_block" in algo_data and "post_function_name" in algo_data:
            s = file_template.render(
                name=algo_data["display_name"],
                image_name=algo_data["name"],
                training_type=algo_data["training_type"],
                skip_pull=self._skip_pull,
                input_dimensionality=algo_data["input_dimensionality"],
                parameters=self._encode_params(algo_data["params"]),
                post_process_block=algo_data["post_process_block"],
                postprocess=algo_data["post_function_name"],
            )
        else:
            s = file_template.render(
                name=algo_data["display_name"],
                image_name=algo_data["name"],
                training_type=algo_data["training_type"],
                skip_pull=self._skip_pull,
                input_dimensionality=algo_data["input_dimensionality"],
                parameters=self._encode_params(algo_data["params"]),
            )
        with target_path.open("w") as fh:
            fh.write(s)

    @staticmethod
    def _encode_params(params: Dict[str, Dict[str, Any]]) -> str:
        s_json = json.dumps(params, sort_keys=True, indent=True)
        return s_json.replace("null", "None").replace("true", "True").replace("false", "False")

    @staticmethod
    def _check_target(target: Union[str, Path],
                      allow_overwrite: bool = False,
                      allow_dir: bool = True,
                      create_parents: bool = False,
                      name: str = "",
                      context: str = "") -> Path:
        target_path = Path(target)
        if target_path.exists():
            if target_path.is_file() and allow_overwrite:
                target_path.unlink()
            elif target_path.is_dir() and allow_dir:
                return AlgorithmGenerator._check_target(target_path / f"{name}.py",
                                                        allow_overwrite=allow_overwrite,
                                                        allow_dir=False,
                                                        context=context)
            else:
                context = f"({context})" if context else f"(for file {name})" if name else ""
                raise ValueError(f"Path '{target}' already exists{context}! "
                                 "Use `force=True` to overwrite existing files.")
        elif allow_overwrite or create_parents:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        return target_path


if __name__ == "__main__":
    test_path = Path("test")
    gen = AlgorithmGenerator("../../../timeeval-algorithms", skip_pull=True)
    # gen.generate_init(test_path / "__init__.py", force=True)
    # gen.generate_algorithm("hybrid_knn", test_path)
    gen.generate_all(test_path, force=True)
