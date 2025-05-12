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

### Known issues

#### Tests with `--dask` fail

- Make sure that password-less SSH is possible from your account, i.a. the following command must work without a password-check:

  ```bash
  ssh localhost
  ```

- You need to install TimeEval properly in the current environment.
  An editable installation (via `pip install -e .`) does not work.

- If it still does not work, make sure that the test files are also installed correctly into the site-packages:

  - Remove the exclusion of tests-files in `setup.py`:

    ```python
    ...
    setup(
        version=VERSION,
        long_description=README,
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests", "tests.*")), # <-- HERE
        url="https://github.com/TimeEval/TimeEval",
    ...
    ```

  - Re-install TimeEval: `pip uninstall timeeval && pip install .`

  - Check that there is a `timeeval`- and a `test`-folder in your installation folder
    (`<somewhere>/lib/python3.<version>/site-packages/`)

- Sometimes there are some connection issues, just run the failing tests again via:

  ```bash
  pytest tests --dask -k <test-name>
  ```
