import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from timeeval_experiments.generator import AlgorithmGenerator


class TestCodeGen(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_path = "tests/example_data/timeeval-algorithms"

    def test_generates_code_correctly(self):
        loader = AlgorithmGenerator(self.repo_path, skip_pull=True)
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            loader.generate_all(tmp_path / "algorithms", force=True)
            shutil.copytree("timeeval", tmp_path / "timeeval")

            process = subprocess.run([
                sys.executable,
                "-c",
                "from algorithms import timeeval_test_algorithm; algo = timeeval_test_algorithm(skip_pull=True); "
                "print(algo.name); print(algo.training_type.value); print(algo.input_dimensionality.value); "
                "print(f\"{algo.main.image_name}:{algo.main.tag}\")"
            ], capture_output=True, check=True, cwd=tmp_path)
            self.assertEqual(
                "DEMO algorithm\nunsupervised\nmultivariate\nmut:5000/akita/timeeval_test_algorithm:latest\n",
                process.stdout.decode("utf-8"),
            )

            process = subprocess.run([
                sys.executable,
                "-c",
                "from algorithms import timeeval_test_algorithm_post; import numpy as np; "
                "algo = timeeval_test_algorithm_post(skip_pull=True); "
                "print(algo.name); "
                "print(list(algo.postprocess(np.arange(3), {})));"
            ], capture_output=True, check=True, cwd=tmp_path)
            self.assertEqual(
                "DEMO algorithm with post-processing\n[0.0, 0.0, 0.0]\n",
                process.stdout.decode("utf-8"),
            )

    def test_target(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path) / "algorithms" / "test"

            target = AlgorithmGenerator._check_target(tmp_path, create_parents=False)
            self.assertEqual(tmp_path, target)
            self.assertFalse(target.parent.exists())

            target = AlgorithmGenerator._check_target(tmp_path, create_parents=True)
            self.assertEqual(tmp_path, target)
            self.assertTrue(target.parent.exists())

    def test_target_folder_exists(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path) / "algorithms"
            tmp_path.mkdir()

            target = AlgorithmGenerator._check_target(tmp_path, allow_overwrite=False, allow_dir=True, name="test")
            self.assertEqual(tmp_path / "test.py", target)

            with self.assertRaises(ValueError):
                AlgorithmGenerator._check_target(tmp_path, allow_overwrite=True, allow_dir=False)

            with self.assertRaises(ValueError):
                AlgorithmGenerator._check_target(tmp_path, allow_overwrite=False, allow_dir=False)

    def test_target_exists(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path) / "test.py"
            tmp_path.touch()

            with self.assertRaises(ValueError):
                AlgorithmGenerator._check_target(tmp_path, allow_overwrite=False)

            # must be at the end, because overwrite=True removes file
            target = AlgorithmGenerator._check_target(tmp_path, allow_overwrite=True)
            self.assertEqual(tmp_path, target)
