# User Guide

## Using TimeEval to evaluate algorithms

```{important}
WIP
```

TimeEval is an evaluation tool for time series anomaly detection algorithms.
We provide a large collection of compatible datasets and algorithms.
The following section describes how you can set up TimeEval to perform your own experiments using the provided datasets and algorithms.
The process consists of three steps: [preparing the datasets](#prepare-datasets), [preparing the algorithms](#prepare-algorithms), and writing the [experiment script](#configure-evaluation-run).

### Prepare datasets

You can download the index file for all our dataset collections from the Datasets website (directly below the table with the dataset download links): datasets.csv.
Place this file at tests/example_data/datasets.csv.

We grouped the datasets into multiple collections because not everybody needs all the datasets, and the whole set is huge.
Each collection can contain a different number of datasets.
The first table on the [_Datasets_ page](https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html) shows you how many datasets are included in each collection, e.g.:

The downloadable ZIP-archives contain the correct folder structure, but your extraction tool might place the contained files into a new folder that is named based on the ZIP-archive-name.
The idea is that you download the index-File (`datasets.csv`) and just the dataset collections that you require, extract them all into the same folder, place the `datasets.csv` there, and use _Option 1_ to select the correct datasets from the folder.

**Example:**

Scenario: You want to use the datasets from the CalIt2 and Daphnet collections.

Dataset download:

```bash
# Download CalIt2.zip, Daphnet.zip and datasets.csv
$ mkdir timeeval-datasets
$ mv datasets.csv timeeval-datasets/
$ unzip CalIt2.zip -d timeeval-datasets
$ unzip Daphnet.zip -d timeeval-datasets
$ tree timeeval-datasets
timeeval-datasets
├── datasets.csv
└── multivariate
    ├── CalIt2
    │   ├── CalIt2-traffic.metadata.json
    │   └── CalIt2-traffic.test.csv
    └── Daphnet
        ├── S01R01E0.metadata.json
        ├── S01R01E0.test.csv
        ├── S01R01E1.metadata.json
        ├── S01R01E1.test.csv
        ├── S01R02E0.metadata.json
        ├── S01R02E0.test.csv
        ├── [...]
        ├── S10R01E1.metadata.json
        └── S10R01E1.test.csv

3 directories, 77 files
```

TimeEval configuration:

```python
dm = MultiDatasetManager([Path("timeeval-datasets")])
datasets = []
datasets.append(dm.select(collection="CalIt2"))
datasets.append(dm.select(collection="Daphnet"))
# ...
```

### Prepare algorithms

### Configure evaluation run

```python
#!/usr/bin/env python3

from pathlib import Path

from timeeval import TimeEval, MultiDatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality, ResourceConstraints
from timeeval.adapters import DockerAdapter
from timeeval.params import FixedParameters
from timeeval.resource_constraints import GB


def main():
    # load datasets and select them
    dm = MultiDatasetManager([Path("datasets")])  # or the path to your datasets folder containing a datasets.csv-index-file
    datasets = dm.select()  # selects ALL available datasets
    # datasets = dm.select(min_anomalies=2)  # select all datasets with at least 2 anomalies
    # we just want 5 datasets for now:
    datasets = datasets[:5]

    # add and configure your algorithms
    algorithms = [Algorithm(
        name="<YOUR ALGORITHM>",
        # set skip_pull=True because the image is already present locally:
        main=DockerAdapter(image_name="<YOUR ALGORITHM IMAGE NAME>", tag="latest", skip_pull=True),
        # the hyperparameters of your algorithm:
        param_config=FixedParameters({
            "window_size": 20,
            "random_state": 42
        }),
        # required by DockerAdapter
        data_as_file=True,
        # UNSUPERVISED --> no training, SEMI_SUPERVISED --> training on normal data, SUPERVISED --> training on anomalies
        # if SEMI_SUPERVISED or SUPERVISED, the datasets must have a corresponding training time series
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.MULTIVARIATE
    )]

    # set the number of repetitions of each algorithm-dataset combination:
    repetitions = 1
    # set resource constraints
    rcs = ResourceConstraints(
        task_memory_limit = 2 * GB,
        task_cpu_limit = 1.0,
    )
    timeeval = TimeEval(dm, datasets, algorithms,
        repetitions=repetitions,
        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC],
        resource_constraints=rcs
    )

    timeeval.run()
    results = timeeval.get_results()
    print(results)

    # detailed results are automatically stored in your current working directory at ./results/<datestring>


if __name__ == "__main__":
    main()
```

## How to integrate your own algorithm into TimeEval

```{important}
WIP
```

If your algorithm is written in Python, you could use our {class}`~timeeval.adapters.function.FunctionAdapter` ([Example](../concepts/algorithms.md#function-adapter) of using the `FunctionAdapter`).
However, this comes with some limitations (such as no way to limit resource usage or setting timeouts).
We, therefore, highly recommend to use the {class}`~timeeval.adapters.docker.DockerAdapter`.
This means that we have to create a Docker image for your algorithm before we can use it in TimeEval.

In the following, we assume that we want to create a Docker image with your algorithm to execute it with TimeEval.
We provide base images for various programming languages.
You can find them [here](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).

```{note}
Please contact the maintainers if there is no base image for your algorithm programming language or runtime.
```

### Procedure

1. Build base image
   1. Clone the timeeval-algorithms repository
   2. Build the selected base image from [`0-base-images`](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).
      Please make sure that you tag your image correctly (the image name must match the `FROM`-clause in your algorithm image.

2. Integrate your algorithm into TimeEval and build the Docker image (you can use any algorithm in this repository as an example for that)
   - TimeEval uses a common interface to execute all its algorithms (using the `DockerAdapter`).
     This interface describes data input and output as well as algorithm configuration.
     The calling-interface is described in this repositories' [README](https://github.com/HPI-Information-Systems/TimeEval-algorithms#timeeval-algorithm-interface).
     Please read the section carefully and adapt your algorithm to the interface description.
     You could also create a wrapper script that takes care of the integration.
     Our canonical file format for time series datasets is described [here](https://github.com/HPI-Information-Systems/TimeEval#canonical-file-format).
   - Create a `Dockerfile` for your algorithm that is based on your selected base image ([example](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/kmeans/Dockerfile)).
   - Build your algorithm Docker image.
   - Check if your algorithm can read a time series using our common file format.
   - Check if the algorithm parameters are correctly set using TimeEval's call format.
   - Check if the anomaly scores are written in the correct format (an anomaly score value for each point of the original time series in a headerless CSV-file).
   - The README contains [example calls](https://github.com/HPI-Information-Systems/TimeEval-algorithms#example-calls) to test your algorithm after you have build the Docker image for it.

3. Install TimeEval (`pip install timeeval==1.2.4`)

4. Create an experiment script with your configuration of datasets, algorithms, etc.

## How to use your own datasets in TimeEval

## Using custom evaluation metrics

## Repetitive runs and measuring runtime

TimeEval has the ability to run an experiment multiple times to improve runtime measurements.
Therefore, {class}`timeeval.TimeEval` has the parameter `repetitions: int = 1`, which tells TimeEval how many times to execute each experiment (algorithm, hyperparameters, and dataset combination).

When measuring runtime, we highly recommend to use TimeEval's feature to limit each algorithm to a specific set of resources (meaning CPU and memory).
This requires using the {class}`timeeval.adapters.docker.DockerAdapter` for the algorithms.


To retrieve the aggregated results, you can call {meth}`timeeval.TimeEval.get_results` with the parameter `aggregated: bool = True`.
This aggregates the quality and runtime measurements  using mean and standard deviation.
Erroneous experiments are excluded from an aggregate.
For example, if you have `repetitions = 5` and one of five experiments failed, the average is built only over the 4 successful runs.

## (Advanced) Distributed execution of TimeEval

```{important}
Before continuing with this guide, please make sure that you have read and understood [this concept page](../concepts/distributed.md).
```

TimeEval uses Dask's SSHCluster to distribute tasks on a compute cluster.
This means that [certain prerequisites](../concepts/distributed.md#cluster-requirements) must be fulfilled before TimeEval can be executed in distributed mode.

We assume that the following requirements are already fulfilled for all hosts of the cluster (independent if the host has the _driver_, _scheduler_, or _worker_ role):

- Python 3 and Docker is installed
- Every node has a virtual environment (Anaconda, virtualenv or similar) with the same name (e.g. `timeeval`) **and prefix**!
- The same TimeEval version is installed in all `timeeval` environments.
- All nodes can reach each other via network (especially via SSH).

Similar to the [local execution of TimeEval](#using-timeeval-to-evaluate-algorithms), we also have to prepare the datasets and algorithms first.

### Prepare time series datasets

1. Please create a dataset folder on each node using the same path. For example: `/data/timeeval-datasets`.
2. Copy your datasets and also the index-file (`datasets.csv`) to all nodes.
3. Test if TimeEval can access this folder and find your datasets on each node:

   ```python
   from timeeval import DatasetManager

   dmgr = DatasetManager("/data/timeeval-datasets", create_if_missing=False)
   dataset = dmgr.get(("<your-collection-name>", "<your-dataset-name>"))
   ```

### Prepare algorithms

If you use plain **Python function**s as algorithm implementations and the {class}`~timeeval.adapters.function.FunctionAdapter`,
please make sure that your Python code is either installed as a module or that the algorithm implementation is part of your single script-file.
Your Python script with the experiment configuration is not allowed to import any other **local** files (e.g., `from .util import xyz`).
This is due to issues with the Python-Path on the remote machines.

If you use **Docker images** for your algorithms and the {class}`~timeeval.adapters.docker.DockerAdapter`,
the algorithm images must be present on all nodes or Docker must be able to pull them from a remote registry (can be controlled with `skip_pull=False`).

There are different ways to get the Docker images to all hosts:

- Build the Docker images locally on each machine (e.g., using a terminal multiplexer)
- Build the Docker images on one machine and distribute them.
  This can be accomplished using image export and import.
  You can follow these rough outline of steps: `docker build`, [`docker image save`](https://docs.docker.com/engine/reference/commandline/image_save/), `rsync` to all machines, [`docker image import`](https://docs.docker.com/engine/reference/commandline/image_import/)
- Push / publish image to a registry available to you (if it's public, you would be responsible for maintaining it)
- [Host your own registry](https://docs.docker.com/registry/introduction/)

At the end, TimeEval must be able to create the algorithms' Docker containers, otherwise it is not able to execute and evaluate them.

### TimeEval configuration for distributed execution

Setting up TimeEval for distributed execution follows the same principles as for [local execution](#using-timeeval-to-evaluate-algorithms).
Two arguments to the {class}`TimeEval-constructor <timeeval.TimeEval>` are relevant for the distributed setup:
`distributed: bool = False` and `remote_config: Optional[RemoteConfiguration] = None`.
You can enable the distributed execution with `distributed=True` and configure the cluster using the {class}`~timeeval.RemoteConfiguration` object.
The following snippet shows the available configuration options:

```python
import sys
from timeeval import RemoteConfiguration

RemoteConfiguration(
    scheduler_host = "localhost",        # scheduler host
    worker_hosts = [],                   # list of worker hosts
    remote_python = sys.executable,      # path to the python executable (same on all hosts)
    dask_logging_file_level = "INFO",    # logging level for the file-based logger
    dask_logging_console_level = "INFO", # logging level for the console logger
    dask_logging_filename = "dask.log",  # filename for the file-based logger used for the Dask-logs
    kwargs_overwrites = {}               # advanced options for DaskSSHCluster
)
```

The _driver_ host (executing TimeEval) must be able to open an SSH connection to all the other nodes using **passwordless SSH**,
otherwise, TimeEval will not be able to reach the other nodes.

If you use resource constraints, please make sure that you set the number of tasks per hosts and the CPU und memory limits correctly.
We highly discourage over-provisioning.
For more details, see the [concept page about resource restrictions](../concepts/configuration.md#resource-restrictions).

## (Advanced) Using hyperparameter heuristics
