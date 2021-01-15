import sys
import pytest
import pathlib
from distutils.cmd import Command
from setuptools import setup, find_packages
from mypy.main import main as mypy

README = (pathlib.Path(__file__).parent / "README.md").read_text(encoding="UTF-8")


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
    long_description=README,
    long_description_content_type="text/markdown",
    author="Phillip Wenig and Sebastian Schmidl",
    author_email="phillip.wenig@hpi.de",
    url="https://gitlab.hpi.de/bp2020fn1/timeeval",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=("tests",)),
    package_data={"timeeval": ["py.typed"]},
    cmdclass={
        "test": PyTestCommand,
        "typecheck": MyPyCheckCommand
    },
    zip_safe=False
)
