# TimeEval

[![pipeline status](https://gitlab.hpi.de/bp2020fn1/timeeval/badges/master/pipeline.svg)](https://gitlab.hpi.de/bp2020fn1/timeeval/-/commits/master)
[![coverage report](https://gitlab.hpi.de/bp2020fn1/timeeval/badges/master/coverage.svg)](https://gitlab.hpi.de/bp2020fn1/timeeval/-/commits/master)

Evaluation Tool for Anomaly Detection Algorithms on Time Series

## Installation using `pip`

Builds of `TimeEval` are published to the [internal package registry](https://gitlab.hpi.de/bp2020fn1/timeeval/-/packages) of the Gitlab instance running at [gitlab.hpi.de](https://gitlab.hpi.de/).

### Prerequisites

- python 3
- pip
- A [personal access token](https://gitlab.hpi.de/help/user/profile/personal_access_tokens.md) with the scope set to `api` for [gitlab.hpi.de](https://gitlab.hpi.de/).
### Steps

You can use `pip` to install TimeEval using:

```sh
pip install TimeEval --extra-index-url https://__token__:<your_personal_token>@gitlab.hpi.de/api/v4/projects/4041/packages/pypi/simple
```

## Installation from source

**tl;dr**

```bash
git clone git@gitlab.hpi.de:bp2020fn1/timeeval.git
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
   Use the file [`environment.yml`](./environment.yml) for this:
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
pytest
```

## Usage

**tl;dr**

```python
from timeeval import TimeEval, Datasets, Algorithm
import numpy as np

# Load dataset metadata
dm = Datasets("data_folder")

# Define algorithm
def my_algorithm(data: np.ndarray) -> np.ndarray:
    return np.zeros_like(data)

# Select datasets and algorithms
datasets = dm.select(collection_name="NAB")
algorithms = [
    # Add algorithms to evaluate...
    Algorithm(
        name="MyAlgorithm",
        function=my_algorithm,
        data_as_file=False
    )
]
timeeval = TimeEval(dm, datasets, algorithms)

# execute evaluation
timeeval.run()

# retrieve results
print(timeeval.results)
```

### Datasets

TimeEval uses a canonical file format for datasets.
Existing datasets in another format must first be transformed into the canonical format before they can be used with TimeEval.

#### Canonical file format

TimeEval's canonical file format is based on CSV.
Each file requires a header, cells (values) are separated by commas (decimal seperator is `.`), and records are separated by newlines (unix-style LF: `\n`).
The first column of the dataset is its index, either in integer- or datetime-format (multiple timestamp-formats are supported but [RFC 3339](https://tools.ietf.org/html/rfc3339) is preferred, e.g. `2017-03-22 15:16:45.433502912`).
The index follows a single or multiple (if multivariate dataset) time series columns.
The last column contains the annotations, `0` for normal points and `1` for anomalies.

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
TimeEval provides a utility to perform this transformation: [`preprocess_datasets.py`](./timeeval/utils/preprocess_dataset.py).

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

[`preprocess_datasets.py`](./timeeval/utils/preprocess_dataset.py) automatically generates the index column using an auto-incrementing integer value.
The integer value can be substituted with a corresponding timestamp (auto-incrementing value is used as a time unit, such as seconds `s` or hours `h` from the unix epoch).
See the tool documentation for further information:

```bash
python timeeval/utils/preprocess_dataset.py --help
```

#### Registering datasets

TimeEval comes with its own collection of benchmark datasets (**currently not included**, find them at `odin01:/home/projects/akita/data/benchmark-data`).
They can directly be used using the dataset manager `Datasets`:

```python
from timeeval import Datasets

dm = Datasets("data_folder")
datasets = dm.select()
```

TimeEval can also use **custom datasets** for the evaluation.
To tell the TimeEval tool where it can find those custom datasets, a configuration file is needed.
The custom datasets config file contains all custom datasets organized by their identifier which is used later on.
Each entry in the config file must contain the path to the dataset and its dedication (usable for training or for testing);
example file `datasets.json`:

```json
{
  "dataset_name": {
    "data": "path/to/data.ts",
     "train_type": "test"
  },
  "other_dataset": {
    "dataset": "dataset2.csv",
     "train_type": "test"
  }
}
```

You can add custom datasets to the dataset manager using two ways:

```python
from timeeval import Datasets

# Directly during initialization
dm = Datasets("data_folder", custom_datasets_file="path/to/custom/datasets.json")

# Later on
dm = Datasets("data_folder")
dm.load_custom_datasets("path/to/custom/datasets.json")
```

### Algorithms

Any algorithm that can be called with a numpy array as parameter and a numpy array as return value can be evaluated. However, so far only __unsupervised__ algorithms are supported.

#### Registering algorithm

```python
from timeeval import TimeEval, Datasets, Algorithm
import numpy as np

def my_algorithm(data: np.ndarray) -> np.ndarray:
    return np.zeros_like(data)

datasets = [("WebscopeS5","A1Benchmark-1")]
algorithms = [
    # Add algorithms to evaluate...
    Algorithm(
        name="MyAlgorithm",
        function=my_algorithm,
        data_as_file=False
    )
]

timeeval = TimeEval(Datasets("data_folder"), datasets, algorithms)
```

### Distributed

TimeEval is able to run multiple tests in parallel on a cluster. It uses [Dask's SSHCluster](https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster) to distribute tasks.
In order to use this feature, the `TimeEval` class accepts a `distributed: bool` flag and additional configurations `ssh_cluster_kwargs: dict` to setup the [SSHCluster](https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster).

