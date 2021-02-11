import sys
import pathlib
from distutils.cmd import Command
from distutils.errors import DistutilsError
from setuptools import setup, find_packages

README = (pathlib.Path(__file__).parent / "README.md").read_text(encoding="UTF-8")


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
    cmdclass={
        "test": PyTestCommand,
        "typecheck": MyPyCheckCommand
    },
    zip_safe=False
)
