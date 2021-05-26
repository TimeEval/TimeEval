import json
import re
import warnings
from pathlib import Path
from typing import Union, Optional, Dict, List

from timeeval_experiments.generator.exceptions import (
    MissingReadmeWarning, MissingManifestWarning, InvalidManifestWarning, AlgorithmManifestLoadingWarning
)

IGNORED_FOLDERS = ["results", "data"]
CODEBLOCK_PATTERN = r"[`]{3}\w*?\n(.+?)[`]{3}"
POST_FUNC_PATTERN = r"def (.+?)\("


def _parse_manifest(algo_dir: Path) -> Optional[Dict]:
    name = algo_dir.name
    manifest_path = algo_dir / "manifest.json"

    if not manifest_path.exists():
        warnings.warn(MissingManifestWarning.msg(name), category=MissingManifestWarning)
        return None

    with manifest_path.open("r") as fh:
        manifest = json.load(fh)
    if "learningType" not in manifest:
        warnings.warn(InvalidManifestWarning.msg(name, "'learningType' is missing."),
                      category=InvalidManifestWarning)
        return None

    return {"display_name": manifest["title"], "training_type": manifest["learningType"]}


def _parse_readme(algo_dir: Path) -> Optional[Dict]:
    name = algo_dir.name
    readme_path = algo_dir / "README.md"

    if not readme_path.exists():
        warnings.warn(MissingReadmeWarning.msg(name), category=MissingReadmeWarning)
        return None

    result = {}
    with readme_path.open("r") as fh:
        lines = "".join(fh.readlines())
    groups = re.findall(CODEBLOCK_PATTERN, lines, re.S)  # G is set through find**all**
    if groups:
        if len(groups) > 1:
            warnings.warn(f"Algorithm {name}'s README contains multiple code blocks, ignoring all of them! "
                          f"If {name} requires post-processing, it will not be generated!",
                          category=AlgorithmManifestLoadingWarning)
        else:
            post_process_block = groups[0]
            matchgroups = re.findall(POST_FUNC_PATTERN, post_process_block, re.M)
            # use first matching group (helper functions are usually placed below the main function)
            result["post_function_name"] = matchgroups[0]
            result["post_process_block"] = post_process_block
    return result


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
