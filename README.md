<div align="center">
<img width="100px" src="https://github.com/TimeEval/TimeEval/raw/main/timeeval-icon.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval</h1>
<p>
Evaluation Tool for Anomaly Detection Algorithms on Time Series.
</p>

[![CI](https://github.com/TimeEval/TimeEval/actions/workflows/build.yml/badge.svg)](https://github.com/TimeEval/TimeEval/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/timeeval/badge/?version=latest)](https://timeeval.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/TimeEval/TimeEval/branch/main/graph/badge.svg?token=esrQJQmMQe)](https://codecov.io/gh/TimeEval/TimeEval)
[![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/TimeEval.svg)](https://badge.fury.io/py/TimeEval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![python version 3.9|3.10|3.11|3.12](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![Downloads](https://pepy.tech/badge/timeeval)](https://pepy.tech/project/timeeval)

</div>

See [TimeEval Algorithms](https://github.com/TimeEval/TimeEval-algorithms) for algorithms that are compatible to this tool.
The algorithms in that repository are containerized and can be executed using the [`DockerAdapter`](./timeeval/adapters/docker.py) of TimeEval.

> If you use TimeEval, please consider [citing](#citation) our paper.

📖 TimeEval's documentation is hosted at https://timeeval.readthedocs.io.

## Features

- Large integrated benchmark dataset collection with more than 700 datasets
- Benchmark dataset interface to select datasets easily
- Adapter architecture for algorithm integration
  - **DockerAdapter**
  - JarAdapter
  - DistributedAdapter
  - MultivarAdapter
  - ... (add your own adapter)
- Large collection of existing algorithm implementations (in [TimeEval Algorithms](https://github.com/TimeEval/TimeEval-algorithms) repository)
- Automatic algorithm detection quality scoring using [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
  (Area under the ROC curve, also _c-statistic_) or range-based metrics
- Automatic timing of the algorithm execution (differentiates pre-, main-, and post-processing)
- Distributed experiment execution
- Output and logfile tracking for subsequent inspection

## Installation

TimeEval can be installed as a package or from source.

> :warning: **Attention!**
>
> Currently, TimeEval is tested **only on Linux and macOS** and relies on unixoid capabilities.
> On Windows, you can use TimeEval within [WSL](https://learn.microsoft.com/windows/wsl/install).
> If you want to use the provided detection algorithms, Docker is required.

### Installation using `pip`

Builds of `TimeEval` are published to [PyPI](https://pypi.org/project/TimeEval/):

#### Prerequisites

- python >= 3.9, < 3.13

  > :warning: **Attention!**
  >
  > A dependency of TimeEval prevents us from supporting Python versions >= 3.13:
  > `prts` is not updated and depends on `NumPy<2.0.0`. However, there is no NumPy
  > version below `2.0.0` that supports `Python>=3.13`.

- pip >= 20

- Docker (for the anomaly detection algorithms)

- (optional) `rsync` for distributed TimeEval

#### Steps

You can use `pip` to install TimeEval from PyPI:

```sh
pip install TimeEval
```

### Installation from source

**tl;dr**

```bash
git clone git@github.com:TimeEval/TimeEval.git
cd timeeval/
conda create -n timeeval python=3.9
conda activate timeeval
pip install .
```

#### Prerequisites

The following tools are required to install TimeEval from source:

- git
- Python > 3.9 and Pip (anaconda or miniconda is preferred)

#### Steps

1. Clone this repository using git and change into its root directory.

2. Create a conda-environment and install all required dependencies:

   ```sh
   conda create -n timeeval python=3.9
   conda activate timeeval
   pip install .
   ```

3. If you want to make changes to TimeEval or run the tests, you need to install the development dependencies with: `pip install ".[ci]"`.
   The optional extra dependencies `".[dev]"` contains additional dependencies for the notebooks and scripts packaged with TimeEval.

## Usage

Example script:

```python
from pathlib import Path
from typing import Dict, Any

import numpy as np

from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
from timeeval.algorithms import subsequence_if
from timeeval.params import FixedParameters

# Load dataset metadata
dm = DatasetManager(Path("tests/example_data"), create_if_missing=False)


# Define algorithm
def my_algorithm(data: np.ndarray, args: Dict[str, Any]) -> np.ndarray:
    score_value = args.get("score_value", 0)
    return np.full_like(data, fill_value=score_value)


# Select datasets and algorithms
datasets = dm.select()
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
    ),
    subsequence_if(params=FixedParameters({"n_trees": 50}))
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
