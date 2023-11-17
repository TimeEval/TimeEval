# Contributor's Guide

## Installation from source

**tl;dr**

```bash
git clone git@github.com:timeeval/timeeval.git
cd timeeval/
conda create -n timeeval python=3.7
conda activate timeeval
pip install -r requirements.txt
python setup.py install
```

### Prerequisites

The following tools are required to install TimeEval from source:

- git
- Python > 3.7 and Pip (anaconda or miniconda is preferred)

### Steps

1. Clone this repository using git and change into its root directory.
2. Create a conda-environment and install all required dependencies.
   ```sh
   conda create -n timeeval python=3.7
   conda activate timeeval
   pip install -r requirements.txt
   ```
3. Build TimeEval:
   `python setup.py bdist_wheel`.
   This should create a Python wheel in the `dist/`-folder.
4. Install TimeEval and all of its dependencies:
   `pip install dist/TimeEval-*-py3-none-any.whl`.
5. If you want to make changes to TimeEval or run the tests, you need to install the development dependencies from `requirements.dev`:
   `pip install -r requirements.dev`.

## Tests

Run tests in `./tests/` as follows

```bash
python setup.py test
```

or

```bash
pytest tests
```

If you want to run the tests that include docker and dask, you need to fulfill some prerequisites:

- Docker is installed and running.
- Your SSH-server is running, and you can SSH to `localhost` with your user without supplying a password.
- You have installed all TimeEval dev dependencies.

You can then run:

```bash
pytest tests --docker --dask
```

### Default Tests

By default, tests that are marked with the following keys are skipped:

- docker
- dask

To run these tests, add the respective keys as parameters: 
```bash
pytest --[key] # e.g. --docker
```
