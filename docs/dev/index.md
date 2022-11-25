# Contributor's Guide

## Installation from source

**tl;dr**

```bash
git clone git@gitlab.hpi.de:akita/bp2020fn1/timeeval.git
cd timeeval/
conda env create --file environment.yml
conda activate timeeval
python setup.py install
```

### Prerequisites

The following tools are required to install TimeEval from source:

- git
- conda (anaconda or miniconda)

### Steps

1. Clone this repository using git and change into its root directory.
2. Create a conda-environment and install all required dependencies.
   Use the file `environment.yml` for this:
   `conda env create --file environment.yml`.
3. Activate the new environment and install TimeEval using _setup.py_:
   `python setup.py install`.
4. If you want to make changes to TimeEval or run the tests, you need to install the development dependencies from `requirements.dev`:
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

If you want to run the tests that include docker and dask, you need to fulfill some prerequesites:

- Docker is installed and running
- Your SSH-server is running, and you can SSH to `localhost` with your users without supplying a password.
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
