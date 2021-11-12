import json
import re
import warnings
from pathlib import Path
from typing import Union, Optional, Dict, List, Any

from .exceptions import (
    MissingReadmeWarning, MissingManifestWarning, InvalidManifestWarning, AlgorithmManifestLoadingWarning
)

IGNORED_FOLDERS = ["results", "data", "scripts"]
CODEBLOCK = r"[`]{3}\w*?\n(.+?)[`]{3}"
CODEBLOCK_PATTERN = re.compile(CODEBLOCK, re.S)  # G is set through find**all**
TE_POST_CODEBLOCK_PATTERN = re.compile(
    r"<!--BEGIN:timeeval-post-->.*" +
    CODEBLOCK +
    r".*<!--END:timeeval-post-->",
    re.S  # G is set through find**all**
)
POST_FUNC_PATTERN = re.compile(r"def (.+?)\(", re.M)


def _parse_manifest(algo_dir: Path) -> Optional[Dict]:
    name = algo_dir.name
    manifest_path = algo_dir / "manifest.json"

    if not manifest_path.exists():
        warnings.warn(MissingManifestWarning.msg(name), category=MissingManifestWarning)
        return None

    with manifest_path.open("r") as fh:
        manifest = json.load(fh)
    if "title" not in manifest:
        warnings.warn(InvalidManifestWarning.msg(name, "'title' is missing."),
                      category=InvalidManifestWarning)
        return None
    if "learningType" not in manifest:
        warnings.warn(InvalidManifestWarning.msg(name, "'learningType' is missing."),
                      category=InvalidManifestWarning)
        return None
    if "inputDimensionality" not in manifest:
        warnings.warn(InvalidManifestWarning.msg(name, "'inputDimensionality' is missing."),
                      category=InvalidManifestWarning)
        return None

    params = _collect_parameters(name, manifest)

    return {
        "display_name": manifest["title"],
        "training_type": manifest["learningType"],
        "input_dimensionality": manifest["inputDimensionality"],
        "params": params
    }


def _collect_parameters(name: str, manifest: Dict[str, Any]) -> Dict:
    parameters = {}
    steps = ["trainingStep", "executionStep"]

    for step in steps:
        if step in manifest:
            exec_params = manifest[step]["parameters"]
            for param_obj in exec_params:
                if param_obj["name"] not in parameters:
                    del param_obj["optional"]
                    parameters[param_obj["name"]] = param_obj

    if len(parameters) == 0:
        warnings.warn(InvalidManifestWarning.msg(name, "no parameters found.", will_skip=False),
                      category=InvalidManifestWarning)

    return parameters


def _parse_readme(algo_dir: Path) -> Optional[Dict]:
    name = algo_dir.name
    readme_path = algo_dir / "README.md"

    if not readme_path.exists():
        warnings.warn(MissingReadmeWarning.msg(name), category=MissingReadmeWarning)
        return None

    result = {}
    with readme_path.open("r") as fh:
        lines = "".join(fh.readlines())
    groups = CODEBLOCK_PATTERN.findall(lines)
    post_func_groups = TE_POST_CODEBLOCK_PATTERN.findall(lines)
    if groups and not post_func_groups:
        warnings.warn(f"Algorithm {name}'s README contains code blocks, but no TimeEval "
                      "post function annotation (fenced code block with "
                      "`<!--BEGIN:timeeval-post-->` and `<!--END:timeeval-post-->`)! "
                      f"If {name} requires post-processing, it will not be generated!",
                      category=AlgorithmManifestLoadingWarning)
    elif post_func_groups:
        post_process_block = post_func_groups[0]
        post_process_block = _fix_indent(post_process_block)
        matchgroups = POST_FUNC_PATTERN.findall(post_process_block)
        # use first matching group (helper functions are usually placed below the main function)
        result["post_function_name"] = matchgroups[0]
        result["post_process_block"] = post_process_block
    return result


def _fix_indent(codeblock: str) -> str:
    lines = codeblock.expandtabs(tabsize=4).split("\n")
    indent_size = len(lines[0]) - len(lines[0].lstrip())
    if indent_size > 0:
        return "\n".join([l[indent_size:] for l in lines])
    else:
        return codeblock


class AlgorithmLoader:
    def __init__(self, timeeval_algorithms_path: Union[str, Path]):
        self.path = Path(timeeval_algorithms_path)
        if not self.path.is_dir():
            raise ValueError(f"{self.path.absolute()} is not a directory!")

        algo_dicts = self._load()
        self._algos = {}
        for a in algo_dicts:
            self._algos[a["name"]] = a

    def _load(self) -> List[Dict]:
        algo_dirs = [
            d for d in self.path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name not in IGNORED_FOLDERS
        ]
        algos = []
        skipped_algos = 0
        for algo_dir in algo_dirs:
            d_manifest = _parse_manifest(algo_dir)
            d_readme = _parse_readme(algo_dir)
            if d_manifest is None or d_readme is None:
                skipped_algos += 1
                continue
            d_algo = {"name": algo_dir.name}
            d_algo.update(d_manifest)
            d_algo.update(d_readme)

            algos.append(d_algo)
        print(f"Found {len(algos)}/{len(algo_dirs)} valid algorithms (skipped {skipped_algos})")
        return algos

    @property
    def algorithm_names(self) -> List[str]:
        return list(self._algos.keys())

    @property
    def all_algorithms(self) -> List[Dict]:
        return list(self._algos.values())

    def algo_detail(self, algo_name: str) -> Dict:
        return self._algos[algo_name]


if __name__ == "__main__":
    loader = AlgorithmLoader("../../../timeeval-algorithms")
    print(loader.algorithm_names)
