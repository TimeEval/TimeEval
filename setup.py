import sys
import pytest
from distutils.cmd import Command
from setuptools import setup, find_packages
from mypy.main import main as mypy


class PyTestCommand(Command):
    description = "run PyTest for TimeEval"
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        pytest.main(["-x", "tests"])


class MyPyCheckCommand(Command):
    description = 'run MyPy for TimeEval; performs static type checking'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        args = ["--pretty", "timeeval"]
        mypy(None, stdout=sys.stdout, stderr=sys.stderr, args=args)


setup(
    name="TimeEval",
    version="0.1.8",
    description="Evaluation Tool for Time Series Anomaly Detection",
    author="Phillip Wenig",
    author_email="phillip.wenig@hpi.de",
    packages=find_packages(),
    package_data={"timeeval": ["py.typed"]},
    cmdclass={
        "test": PyTestCommand,
        "typecheck": MyPyCheckCommand
    },
    zip_safe=False
)
