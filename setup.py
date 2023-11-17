import glob
import os
import shutil
import sys
from pathlib import Path

from setuptools import setup, find_packages, Command


README = (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")
HERE = Path(os.path.dirname(__file__)).absolute()
# get __version__ from timeeval/_version.py
with open(Path("timeeval") / "_version.py") as f:
    exec(f.read())
VERSION: str = __version__  # noqa


def load_dependencies():
    EXCLUDES = ["python", "pip"]
    with open(HERE / "requirements.txt", "r", encoding="UTF-8") as f:
        env = f.readlines()

    def excluded(name):
        return any([excl in name for excl in EXCLUDES])

    deps = [dep for dep in env if not excluded(dep)]
    return deps


class PyTestCommand(Command):
    description = "run PyTest for TimeEval"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        import pytest
        from pytest import ExitCode

        exit_code = pytest.main(["--cov-report=term", "--cov-report=xml:coverage.xml", "--cov=timeeval",
                                 "--cov=timeeval_experiments.generator", "--optuna", "tests"])
        if exit_code == ExitCode.TESTS_FAILED:
            raise ValueError("Tests failed!")
        elif exit_code == ExitCode.INTERRUPTED:
            raise ValueError("pytest was interrupted!")
        elif exit_code == ExitCode.INTERNAL_ERROR:
            raise ValueError("pytest internal error!")
        elif exit_code == ExitCode.USAGE_ERROR:
            raise ValueError("Pytest was not correctly used!")
        elif exit_code == ExitCode.NO_TESTS_COLLECTED:
            raise ValueError("No tests found!")
        # else: everything is fine


class MyPyCheckCommand(Command):
    description = 'run MyPy for TimeEval; performs static type checking'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from mypy.main import main as mypy

        args = ["--pretty", "timeeval", "timeeval_experiments"]
        mypy(None, stdout=sys.stdout, stderr=sys.stderr, args=args)


class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        files = [
            ".coverage*",
            "coverage.xml"
        ]
        dirs = ["build", "dist", "*.egg-info", "**/__pycache__", ".mypy_cache",
                ".pytest_cache", "**/.ipynb_checkpoints"]
        for d in dirs:
            for filename in glob.glob(d):
                shutil.rmtree(filename, ignore_errors=True)

        for f in files:
            for filename in glob.glob(f):
                try:
                    os.remove(filename)
                except OSError:
                    pass


if __name__ == "__main__":
    setup(
        name="TimeEval",
        version=VERSION,
        description="Evaluation Tool for Time Series Anomaly Detection Methods",
        long_description=README,
        long_description_content_type="text/markdown",
        author="Phillip Wenig and Sebastian Schmidl",
        author_email="phillip.wenig@hpi.de",
        url="https://github.com/TimeEval/TimeEval",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9"
        ],
        packages=find_packages(exclude=("tests", "tests.*")),
        package_data={"timeeval": ["py.typed"], "timeeval_experiments": ["py.typed"]},
        install_requires=load_dependencies(),
        python_requires=">=3.7, <3.10",
        test_suite="tests",
        cmdclass={
            "test": PyTestCommand,
            "typecheck": MyPyCheckCommand,
            "clean": CleanCommand
        },
        zip_safe=False
    )
