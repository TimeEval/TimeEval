import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from timeeval_experiments.generator import AlgorithmLoader
from timeeval_experiments.generator.algorithm_parsing import _parse_readme, _parse_manifest
from timeeval_experiments.generator.exceptions import MissingReadmeWarning, MissingManifestWarning, \
    InvalidManifestWarning, AlgorithmManifestLoadingWarning


class TestAlgorithmParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_path = "tests/example_data/timeeval-algorithms"
        self.invalid_manifest_1 = {"title": "DEMO algorithm"}
        self.invalid_manifest_2 = {"learningType": "unsupervised"}

    def test_parses_fstree_correctly(self):
        loader = AlgorithmLoader(self.repo_path)
        self.assertListEqual(loader.algorithm_names, ["timeeval_test_algorithm_post", "timeeval_test_algorithm"])
        algos = loader.all_algorithms
        self.assertEqual(len(algos), 2)
        self.assertDictEqual(algos[0], {
            "display_name": "DEMO algorithm with post-processing",
            "name": "timeeval_test_algorithm_post",
            "training_type": "unsupervised",
            "post_function_name": "post_func",
            "post_process_block": "import numpy as np\ndef post_func(X, args):\n    return np.zeros(X.shape[0])\n"
        })
        self.assertDictEqual(algos[1], {
            "display_name": "DEMO algorithm",
            "name": "timeeval_test_algorithm",
            "training_type": "unsupervised"
        })
        self.assertEqual(algos[1], loader.algo_detail("timeeval_test_algorithm"))

    def test_skips_missing_readme(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            algorithm_folder = Path(tmp_path) / "algorithm"
            algorithm_folder.mkdir()
            with self.assertWarns(MissingReadmeWarning):
                res = _parse_readme(algorithm_folder)
                self.assertIsNone(res)

    def test_ignores_invalid_readme(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            algorithm_folder = Path(tmp_path) / "algorithm"
            algorithm_folder.mkdir()
            with (algorithm_folder / "README.md").open("w") as fh:
                fh.write(
                    """# Demo broken postprocess Readme
                    ```python
                    import numpy
                    ```
                    test bla bla
                    ```python
                    class Main:
                        pass
                    ```"""
                )
            with self.assertWarns(AlgorithmManifestLoadingWarning):
                res = _parse_readme(algorithm_folder)
                self.assertDictEqual(res, {})

    def test_skips_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            algorithm_folder = Path(tmp_path) / "algorithm"
            algorithm_folder.mkdir()
            with self.assertWarns(MissingManifestWarning):
                res = _parse_manifest(algorithm_folder)
                self.assertIsNone(res)

    def test_skips_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            algorithm_folder = Path(tmp_path) / "algorithm"
            algorithm_folder.mkdir()
            with (algorithm_folder / "manifest.json").open("w") as fh:
                json.dump(self.invalid_manifest_1, fh)
            with self.assertWarns(InvalidManifestWarning):
                res = _parse_manifest(algorithm_folder)
                self.assertIsNone(res)

            with (algorithm_folder / "manifest.json").open("w") as fh:
                json.dump(self.invalid_manifest_2, fh)
            with self.assertWarns(InvalidManifestWarning):
                res = _parse_manifest(algorithm_folder)
                self.assertIsNone(res)
