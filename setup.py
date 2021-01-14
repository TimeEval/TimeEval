from distutils.cmd import Command
from setuptools import setup, find_packages
import pytest


class PyTestCommand(Command):
    description = 'run PyTest for TimeEval'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        pytest.main(["-x", "tests"])


setup(
    name="TimeEval",
    version="0.1.8",
    description="Evaluation Tool for Time Series Anomaly Detection",
    author="Phillip Wenig",
    author_email="phillip.wenig@hpi.de",
    packages=find_packages(),
    cmdclass={
        'test': PyTestCommand
    }
)
