<div align="center">
<img width="100px" src="./timeeval-icon.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval</h1>
<p>
Evaluation Tool for Anomaly Detection Algorithms on time series.
</p>

![pipeline status](https://gitlab.hpi.de/akita/timeeval/badges/main/pipeline.svg)
![coverage report](https://gitlab.hpi.de/akita/timeeval/badges/main/coverage.svg)
[![PyPI version](https://badge.fury.io/py/TimeEval.svg)](https://badge.fury.io/py/TimeEval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

</div>

See [TimeEval Algorithms](https://gitlab.hpi.de/akita/timeeval-algorithms) (use [this link](https://github.com/HPI-Information-Systems/TimeEval-algorithms) on Github) for algorithms that are compatible to this tool.
The algorithms in this repository are containerized and can be executed using the [`DockerAdapter`](./timeeval/adapters/docker.py) of TimeEval.

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
- (optional) A [personal access token](https://gitlab.hpi.de/help/user/profile/personal_access_tokens.md) with the scope set to `api` (read) or another type of access token able to read the package registry of TimeEval hosted at [gitlab.hpi.de](https://gitlab.hpi.de/).

#### Steps

You can use `pip` to install TimeEval using (PyPI):

```sh
pip install TimeEval
```

or (PI@gitlab.hpi.de):

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
from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
from timeeval.constants import HPI_CLUSTER
import numpy as np


# Load dataset metadata
dm = DatasetManager(HPI_CLUSTER.akita_benchmark_path)


# Define algorithm
def my_algorithm(data: np.ndarray) -> np.ndarray:
    return np.zeros_like(data)


# Select datasets and algorithms
datasets = dm.select(collection="NAB")
algorithms = [
    # Add algorithms to evaluate...
    Algorithm(
        name="MyAlgorithm",
        main=FunctionAdapter(my_algorithm),
        data_as_file=False,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.UNIVARIATE,
    )
]
timeeval = TimeEval(dm, datasets, algorithms)

# execute evaluation
timeeval.run()

# retrieve results
print(timeeval.get_results())
```

### Datasets

TimeEval uses a canonical file format for datasets.
Existing datasets in another format must first be transformed into the canonical format before they can be used with TimeEval.

#### Canonical file format

TimeEval's canonical file format is based on CSV.
Each file requires a header, cells (values) are separated by commas (decimal seperator is `.`), and records are separated by newlines (unix-style LF: `\n`).
The first column of the dataset is its index, either in integer- or datetime-format
(multiple timestamp-formats are supported but [RFC 3339](https://tools.ietf.org/html/rfc3339) is preferred, e.g. `2017-03-22 15:16:45.433502912`).
The index follows a single or multiple (if multivariate dataset) time series columns.
The last column contains the annotations, `0` for normal points and `1` for anomalies.
Usage of the `timestamp` and `is_anomaly` column headers is recommended.

```csv
timestamp,value,is_anomaly
0,12751.0,1
1,8767.0,0
2,7005.0,0
3,5257.0,0
4,4189.0,0
```

#### Dataset preprocessing

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

#### Registering datasets

TimeEval comes with its own collection of benchmark datasets (**currently not included**, download them [from our website](https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html)).
They can directly be used using the dataset manager `DatasetManager`:

```python
from pathlib import Path

from timeeval import DatasetManager
from timeeval.constants import HPI_CLUSTER

datasets_folder: Path = HPI_CLUSTER.akita_benchmark_path  # or Path("./datasets-folder")
dm = DatasetManager(datasets_folder)
datasets = dm.select()
```

##### Custom datasets

TimeEval can also use **custom datasets** for the evaluation.
The time series CSV-files must still follow our canonical file format!

To tell the TimeEval tool where it can find those custom datasets, a configuration file is needed.
The custom datasets config file contains all custom datasets organized by their identifier which is used later on.
Each entry in the config file must contain the path to the test time series;
optionally, one can add a path to the training time series, specify the dataset type, and supply the period size if known.
The paths to the data files must be absolute or relative to the configuration file.
Example file `custom_datasets.json`:

```json
{
  "dataset_name": {
    "test_path": "/absolute/path/to/data.csv"
  },
  "other_supervised_dataset": {
    "test_path": "/absolute/path/to/test.csv",
    "train_path": "./train.csv",
    "type": "synthetic",
    "period": 20
  }
}
```

You can add custom datasets to the dataset manager using two ways:

```python
from pathlib import Path

from timeeval import DatasetManager
from timeeval.constants import HPI_CLUSTER

custom_datasets_path = Path("/absolute/path/to/custom_datasets.json")

# Directly during initialization
dm = DatasetManager(HPI_CLUSTER.akita_benchmark_path, custom_datasets_file=custom_datasets_path)

# Later on
dm = DatasetManager(HPI_CLUSTER.akita_benchmark_path)
dm.load_custom_datasets(custom_datasets_path)
```

### Algorithms

Any algorithm that can be called with a numpy array as parameter and a numpy array as return value can be evaluated.
TimeEval also supports passing only the filepath to an algorithm and let the algorithm perform the file reading and parsing.
In this case, the algorithm must be able to read to data format described [earlier](#Canonical-file-format).
Use `data_as_file=True` as a keyword argument to the algorithm declaration.

The `main` function of an algorithm must implement the `timeeval.adapters.Adapter`-interface.
TimeEval comes with four different adapter types described in section [Algorithm adapters](#Algorithm-adapters).

Each algorithm is associated with metadata including its learning type and input dimensionality.
TimeEval distinguishes between the three learning types `LearningType.UNSUPERVISED` (default), `LearningType.SEMI_SUPERVISED`, and `LearningType.SUPERVISED`
and the two input dimensionality definitions `InputDimensionality.UNIVARIATE` (default) and `InputDimensionality.MULTIVARIATE`.

#### Registering algorithms

```python
from timeeval import TimeEval, DatasetManager, Algorithm
from timeeval.adapters import FunctionAdapter
from timeeval.constants import HPI_CLUSTER
import numpy as np

def my_algorithm(data: np.ndarray) -> np.ndarray:
    return np.zeros_like(data)

datasets = [("WebscopeS5", "A1Benchmark-1")]
algorithms = [
    # Add algorithms to evaluate...
    Algorithm(
        name="MyAlgorithm",
        main=FunctionAdapter(my_algorithm),
        data_as_file=False,
    )
]

timeeval = TimeEval(DatasetManager(HPI_CLUSTER.akita_benchmark_path), datasets, algorithms)
```

#### Algorithm adapters

Algorithm adapters allow you to use different algorithm types within TimeEval.
The most basic adapter just wraps a python-function.

You can implement your own adapters.
Example:

```python
from typing import Optional
from timeeval.adapters.base import Adapter
from timeeval.data_types import AlgorithmParameter


class MyAdapter(Adapter):

    # AlgorithmParameter = Union[np.ndarray, Path]
    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        # e.g. create another process or make a call to another language
        pass
```

##### Function adapter

The [`FunctionAdapter`](./timeeval/adapters/function.py) allows you to use Python functions and methods as the algorithm
main code.
You can use this adapter by wrapping your function:

```python
from timeeval import Algorithm
from timeeval.adapters import FunctionAdapter
from timeeval.data_types import AlgorithmParameter
import numpy as np

def your_function(data: AlgorithmParameter, args: dict) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.zeros_like(data)
    else: # data = pathlib.Path
        return np.genfromtxt(data)[0]

Algorithm(
    name="MyPythonFunctionAlgorithm",
    main=FunctionAdapter(your_function),
    data_as_file=False
)
```

##### Distributed adapter

The [`DistributedAdapter`](./timeeval/adapters/distributed.py) allows you to execute an already distributed algorithm on multiple machines.
Supply a list of `remote_hosts` and a `remote_command` to this adapter.
It will use SSH to connect to the remote hosts and execute the `remote_command` on these hosts before starting the main algorithm locally. 

> **Attention!**
>
> - Password-less ssh to the remote machines required!
> - **Do not combine with the distributed execution of TimeEval ("TimeEval.Distributed" using `TimeEval(..., distributed=True)`)!**
>   This will affect the timing results.

##### Jar adapter

The [`JarAdapter`](./timeeval/adapters/distributed.py) lets you evaluate Java algorithms in TimeEval.
You can supply the path to the Jar-File (executable) and any additional arguments to the Java-process call.

##### Adapter to apply univariate methods to multivariate data

The [`MultivarAdapter`](./timeeval/adapters/distributed.py) allows you to apply an univariate algorithm to each dimension of a multivariate dataset individually
and receive a single aggregated result.
You can currently choose between three different result aggregation strategies that work on single points:

- `timeeval.adapters.multivar.AggregationMethod.MEAN`
- `timeeval.adapters.multivar.AggregationMethod.MEDIAN`
- `timeeval.adapters.multivar.AggregationMethod.MAX`

If `n_jobs > 1`, the algorithms are executed in parallel.

#### Docker adapter

The [`DockerAdapter`](./timeeval/adapters/docker.py) allows you to run an algorithm as a Docker container.
This means that the algorithm is available as a Docker image.
This is the main adapter used for our evaluations.
Usage example:

```python
from timeeval import Algorithm
from timeeval.adapters import DockerAdapter

Algorithm(
    name="MyDockerAlgorithm",
    main=DockerAdapter(image_name="algorithm-docker-image", tag="latest"),
    data_as_file=True  # important here!
)
```

> **Attention!**
>
> Using a `DockerAdapter` implies that `data_as_file=True` in the `Algorithm` construction.
> The adapter supplies the dataset to the algorithm via bind-mounting and does not support passing the data as numpy array.

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

## Timeout Algorithms consuming too much time

Some algorithms are not suitable for very large datasets and, thus, can take a long time until they finish. Therefore, the `DockerAdapter` class can take in a `timeout` parameter that defines the maximum amount of time the algorithm is allowed to run. The parameter takes in a `durations.Duration` object. If the timeout is exceeded, a `DockerTimeoutError` is raised and the specific algorithm for the current dataset is canceled.
