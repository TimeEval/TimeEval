<div align="center">
<img width="100px" src="https://github.com/HPI-Information-Systems/TimeEval/raw/main/timeeval-icon.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval</h1>
<p>
Evaluation Tool for Anomaly Detection Algorithms on Time Series.
</p>

[![CI](https://github.com/HPI-Information-Systems/TimeEval/actions/workflows/build.yml/badge.svg)](https://github.com/HPI-Information-Systems/TimeEval/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/timeeval/badge/?version=latest)](https://timeeval.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/HPI-Information-Systems/TimeEval/branch/main/graph/badge.svg?token=esrQJQmMQe)](https://codecov.io/gh/HPI-Information-Systems/TimeEval)
[![PyPI version](https://badge.fury.io/py/TimeEval.svg)](https://badge.fury.io/py/TimeEval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

</div>

See [TimeEval Algorithms](https://gitlab.hpi.de/akita/timeeval-algorithms) (use [this link](https://github.com/HPI-Information-Systems/TimeEval-algorithms) on Github) for algorithms that are compatible to this tool.
The algorithms in this repository are containerized and can be executed using the [`DockerAdapter`](./timeeval/adapters/docker.py) of TimeEval.

> If you use TimeEval, please consider [citing](#citation) our paper.

📖 TimeEval's documentation is hosted at https://timeeval.readthedocs.io.

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

You can use `pip` to install TimeEval from PyPI:

```sh
pip install TimeEval
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
dm = DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK], create_if_missing=False)

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

## Citation

If you use TimeEval in your project or research, please cite our demonstration paper:

> Phillip Wenig, Sebastian Schmidl, and Thorsten Papenbrock.
> TimeEval: A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms. PVLDB, 15(12): 3678 - 3681, 2022.
> doi:[10.14778/3554821.3554873](https://doi.org/10.14778/3554821.3554873)

```bibtex
@article{WenigEtAl2022TimeEval,
  title = {TimeEval: {{A}} Benchmarking Toolkit for Time Series Anomaly Detection Algorithms},
  author = {Wenig, Phillip and Schmidl, Sebastian and Papenbrock, Thorsten},
  date = {2022},
  journaltitle = {Proceedings of the {{VLDB Endowment}} ({{PVLDB}})},
  volume = {15},
  number = {12},
  pages = {3678--3681},
  doi = {10.14778/3554821.3554873}
}
```
