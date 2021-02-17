import sys
import os
from pathlib import Path
from distutils.cmd import Command
from distutils.errors import DistutilsError
from setuptools import setup, find_packages

try:
    import yaml
except ImportError:
    import pip
    pip.main(["install", "pyyml"])
    import yaml


README = (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")
HERE = Path(os.path.dirname(__file__)).absolute()


def load_dependencies():
    EXCLUDES = ["python"]
    with open(HERE / "environment.yml", "r", encoding="UTF-8") as f:
        env = yaml.safe_load(f)

    def split_deps(deps):
        pip = list(filter(lambda x: isinstance(x, dict), deps))
        if len(pip) == 1:
            pip = pip[0].get("pip", []) or []
        conda = list(filter(lambda x: not isinstance(x, dict), deps))
        return pip, conda

    def to_pip(dep):
        parts = dep.split("=")
        if len(parts) == 1:
            return dep
        else:
            return "==".join(parts)

    def excluded(name):
        return any([excl in name for excl in EXCLUDES])

    pip_deps, conda_deps = split_deps(env.get("dependencies", []))
    conda_deps = [to_pip(dep) for dep in conda_deps if not excluded(dep)]
    return conda_deps + pip_deps


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

        exit_code = pytest.main(["--cov=timeeval", "-x", "tests"])
        if exit_code == ExitCode.TESTS_FAILED:
            raise DistutilsError("Tests failed!")
        elif exit_code == ExitCode.INTERRUPTED:
            raise DistutilsError("pytest was interrupted!")
        elif exit_code == ExitCode.INTERNAL_ERROR:
            raise DistutilsError("pytest internal error!")
        elif exit_code == ExitCode.USAGE_ERROR:
            raise DistutilsError("Pytest was not correctly used!")
        elif exit_code == ExitCode.NO_TESTS_COLLECTED:
            raise DistutilsError("No tests found!")
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

        args = ["--pretty", "timeeval", "tests"]
        mypy(None, stdout=sys.stdout, stderr=sys.stderr, args=args)


setup(
    name="TimeEval",
    version="0.3.1",
    description="Evaluation Tool for Time Series Anomaly Detection",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Phillip Wenig and Sebastian Schmidl",
    author_email="phillip.wenig@hpi.de",
    url="https://gitlab.hpi.de/akita/bp2020fn1/timeeval",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=("tests",)),
    package_data={"timeeval": ["py.typed"]},
    install_requires=load_dependencies(),
    python_requires=">=3.7",
    cmdclass={
        "test": PyTestCommand,
        "typecheck": MyPyCheckCommand
    },
    zip_safe=False
)
