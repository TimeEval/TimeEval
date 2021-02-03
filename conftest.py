import pytest


options = {"docker": "mark test to run with docker installed",
           "dask": "mark test to run with dask installed and being able to SSH itself."}


def pytest_addoption(parser):
    for option in options.keys():
        parser.addoption(
            f"--{option}", action="store_true", default=False, help=f"run also {option} tests"
        )


def pytest_configure(config):
    for option, desc in options.items():
        config.addinivalue_line("markers", f"{option}: {desc} ")


def pytest_collection_modifyitems(config, items):
    for option in options.keys():
        if config.getoption(f"--{option}"):
            continue
        skip = pytest.mark.skip(reason=f"need --{option} option to run")
        for item in items:
            if option in item.keywords:
                item.add_marker(skip)
