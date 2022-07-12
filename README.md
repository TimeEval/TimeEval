<div align="center">
<img width="100px" src="https://github.com/HPI-Information-Systems/TimeEval/raw/main/timeeval-icon.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval</h1>
<p>
Evaluation Tool for Anomaly Detection Algorithms on Time Series.
</p>

[![CI](https://github.com/HPI-Information-Systems/TimeEval/actions/workflows/test-build-matrix.yml/badge.svg)](https://github.com/HPI-Information-Systems/TimeEval/actions/workflows/test-build-matrix.yml)
[![Documentation Status](https://readthedocs.org/projects/timeeval/badge/?version=latest)](https://timeeval.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/HPI-Information-Systems/TimeEval/branch/main/graph/badge.svg?token=esrQJQmMQe)](https://codecov.io/gh/HPI-Information-Systems/TimeEval)
[![PyPI version](https://badge.fury.io/py/TimeEval.svg)](https://badge.fury.io/py/TimeEval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

</div>

See [TimeEval Algorithms](https://gitlab.hpi.de/akita/timeeval-algorithms) (use [this link](https://github.com/HPI-Information-Systems/TimeEval-algorithms) on Github) for algorithms that are compatible to this tool.
The algorithms in this repository are containerized and can be executed using the [`DockerAdapter`](./timeeval/adapters/docker.py) of TimeEval.

> If you use TimeEval, please consider [citing](#citation) our paper.

## Features

- Large integrated benchmark dataset collection with more than 700 datasets
- Benchmark dataset interface to select datasets easily
- Adapter architecture for algorithm integration
  - JarAdapter
  - DistributedAdapter
  - MultivarAdapter
  - DockerAdapter
  - ... (add your own adapter)
- Automatic algorithm detection quality scoring using [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) (Area under the ROC curve, also _c-statistic_) metric
- Automatic timing of the algorithm execution (differentiates pre-, main-, and post-processing)
- Distributed experiment execution
- Output and logfile tracking for subsequent inspection

## Mechanics

TimeEval takes your input and automatically creates experiment configurations by taking the cross-product of your inputs.
It executes all experiments configuration one after the other or - when distributed - in parallel and records the anomaly detection quality and the runtime of the algorithms.

TimeEval takes 4 different inputs for the experiment creation:

- Algorithms
- Datasets
- Algorithm ParameterGrids
- A repetition number

### TimeEval.Distributed

TimeEval is able to run multiple tests in parallel on a cluster. It uses [Dask's SSHCluster](https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster) to distribute tasks.
In order to use this feature, the `TimeEval` class accepts a `distributed: bool` flag and additional configurations `ssh_cluster_kwargs: dict` to setup the [SSHCluster](https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster).

### Repetitive runs and scoring

TimeEval has the ability to run an experiment multiple times.
Therefore, the `TimeEval` class has the parameter `repetitions: int = 1`.
Each algorithm on every dataset is run `repetitions` times.
To retrieve the aggregated results, the `TimeEval` class provides the method `get_results` which wants to know whether the results should be `aggregated: bool = True`.
Erroneous experiments are excluded from an aggregate.
For example, if you have `repetitions = 5` and one of five experiments failed, the average is built only over the 4 successful runs.
To retrieve the raw results, you can either `timeeval.get_results(aggregated=False)` or call the results object directly: `timeeval.results`.

## Installation

TimeEval can be installed as a package or from source.

### Installation using `pip`

Builds of `TimeEval` are published to the [internal package registry](https://gitlab.hpi.de/akita/timeeval/-/packages) of the Gitlab instance running at [gitlab.hpi.de](https://gitlab.hpi.de/) and to [PyPI](https://pypi.org/project/TimeEval/).

#### Prerequisites

- python >= 3.7, <=3.9
- pip >= 20
- Docker
- (optional) A [personal access token](https://gitlab.hpi.de/help/user/profile/personal_access_tokens.md) with the scope set to `api` (read) or another type of access token able to read the package registry of TimeEval hosted at [gitlab.hpi.de](https://gitlab.hpi.de/).
- (optional) `rsync` for distributed TimeEval

#### Steps

You can use `pip` to install TimeEval using (PyPI):

```sh
pip install TimeEval
```

or (Package Index @ gitlab.hpi.de):

```sh
pip install TimeEval --extra-index-url https://__token__:<your_personal_token>@gitlab.hpi.de/api/v4/projects/4041/packages/pypi/simple
```

### Installation from source

**tl;dr**

```bash
git clone git@gitlab.hpi.de:akita/bp2020fn1/timeeval.git
cd timeeval/
conda env create --file environment.yml
conda activate timeeval
python setup.py install
```

#### Prerequisites

The following tools are required to install TimeEval from source:

- git
- conda (anaconda or miniconda)

#### Steps

1. Clone this repository using git and change into its root directory.
2. Create a conda-environment and install all required dependencies.
   Use the file [`environment.yml`](./environment.yml) for this:
   `conda env create --file environment.yml`.
3. Activate the new environment and install TimeEval using _setup.py_:
   `python setup.py install`.
4. If you want to make changes to TimeEval or run the tests, you need to install the development dependencies from `requirements.dev`:
   `pip install -r requirements.dev`.

## Usage

**tl;dr**

```python
from typing import Dict, Any

import numpy as np

from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
from timeeval.constants import HPI_CLUSTER
from timeeval.params import FixedParameters


# Load dataset metadata
dm = DatasetManager(HPI_CLUSTER.akita_benchmark_path, create_if_missing=False)

# Define algorithm
def my_algorithm(data: np.ndarray, args: Dict[str, Any]) -> np.ndarray:
    score_value = args.get("score_value", 0)
    return np.full_like(data, fill_value=score_value)

# Select datasets and algorithms
datasets = dm.select(collection="NAB")
datasets = datasets[-1:]
# Add algorithms to evaluate...
algorithms = [
    Algorithm(
        name="MyAlgorithm",
        main=FunctionAdapter(my_algorithm),
        data_as_file=False,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.UNIVARIATE,
        param_config=FixedParameters({"score_value": 1.})
    )
]
timeeval = TimeEval(dm, datasets, algorithms)

# execute evaluation
timeeval.run()
# retrieve results
print(timeeval.get_results())
```

### Dataset preprocessing

Datasets in different formats should be transformed in TimeEval's canonical file format.
TimeEval provides a utility to perform this transformation: [`preprocess_datasets.py`](scripts/preprocess_dataset.py).

A single dataset can be provided in two Numpy-readable text files.
The first text file contains the data.
The labels must be in a separate text file.
Hereby, the label file can either contain the actual labels for each point in the data file or only the line indices of the anomalies.
Example source data files:

Data file

```csv
12751.0
8767.0
7005.0
5257.0
4189.0
```

Labels file (actual labels)

```csv
1
0
0
0
0
```

Labels file (line indices)

```csv
3
4
```

[`preprocess_datasets.py`](scripts/preprocess_dataset.py) automatically generates the index column using an auto-incrementing integer value.
The integer value can be substituted with a corresponding timestamp (auto-incrementing value is used as a time unit, such as seconds `s` or hours `h` from the unix epoch).
See the tool documentation for further information:

```bash
python timeeval/utils/preprocess_dataset.py --help
```

## Tests

Run tests in `./tests/` as follows

```bash
python setup.py test
```

or

```bash
pytest
```

### Default Tests

By default, tests that are marked with the following keys are skipped:

- docker
- dask

To run these tests, add the respective keys as parameters: 
```bash
pytest --[key] # e.g. --docker
```

## Use a time limit to restrict runtime of long-running Algorithms

Some algorithms are not suitable for very large datasets and, thus, can take a long time until they finish either training or testing.
For this reason, TimeEval uses timeouts to restrict the runtime of all algorithms.
You can change the timeout values for the training and testing phase globally using configuration options in the `ResourceConstraints` class:

```python
from durations import Duration
from timeeval import TimeEval, ResourceConstraints

limits = ResourceConstraints(
    train_timeout=Duration("2 hours"),
    execute_timeout=Duration("2 hours"),
)
timeeval = TimeEval(dm, datasets, algorithms, resource_constraints=limits)
...
```

> **Attention!**
> 
> Currently, only the `DockerAdapter`-class can deal with resource constraints.
> All other adapters ignore them.

It's also possible to use different timeouts for specific algorithms if they run using the `DockerAdapter`.
The `DockerAdapter` class can take in a `timeout` parameter that defines the maximum amount of time the algorithm is allowed to run.
The parameter takes in a `durations.Duration` object as well and overwrites the globally set timeouts.
If the timeout is exceeded, a `DockerTimeoutError` is raised and the specific algorithm for the current dataset is cancelled.

## Citation

If you use TimeEval in your project or research, please cite our demonstration paper:

> Phillip Wenig, Sebastian Schmidl, and Thorsten Papenbrock.
> TimeEval: A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms. PVLDB, 15(12): XXXX - XXXX, 2022.
> doi:[YYYY](https://doi.org/YYYY)
>
> _To appear in [PVLDB 2022 volume 15 issue 12](https://vldb.org/2022/)_.

```bibtex
@article{WenigEtAl2022TimeEval,
  title = {TimeEval: {{A}} Benchmarking Toolkit for Time Series Anomaly Detection Algorithms},
  author = {Wenig, Phillip and Schmidl, Sebastian and Papenbrock, Thorsten},
  date = {2022},
  journaltitle = {Proceedings of the {{VLDB Endowment}} ({{PVLDB}})},
  volume = {15},
  number = {12},
  pages = {XXXX--XXXX},
  doi = {YYYY}
}
```
