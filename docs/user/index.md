# User Guide

This part of the TimeEval documentation includes a couple of usage guides to get you started on using TimeEval for your own projects.
The guides teach you TimeEval's APIs and their usage, but they do not get into detail about how TimeEval works.
You can find the detailed descriptions of TimeEval concepts [here](../concepts/index.md).

```{toctree}
---
maxdepth: 2
---
.
```

## Using TimeEval to evaluate algorithms

TimeEval is an evaluation tool for time series anomaly detection algorithms.
We provide a large collection of compatible datasets and algorithms.
The following section describes how you can set up TimeEval to perform your own experiments using the provided datasets and algorithms.
The process consists of three steps: [preparing the datasets](#prepare-datasets), [preparing the algorithms](#prepare-algorithms), and writing the [experiment script](#configure-evaluation-run).

### Prepare datasets

This section assumes that you want to use the TimeEval datasets.
If you want to use your own datasets with TimeEval, please read [](#how-to-integrate-your-own-algorithm-into-timeeval).

For the evaluation of time series anomaly detection algorithms, we collected univariate and multivariate time series datasets from various sources.
We looked out for real-world as well as synthetically generated datasets with real-valued values and anomaly annotations.
We included datasets with direct anomaly annotations (points or subsequences are labelled as either normal (0) or anomalous (1)) and indirect anomaly annotations.
For the later, we included datasets with categorical labels, where a single class (or low number of classes) is clearly underrepresented and can be assigned to unwanted, erroneous, or anomalous behavior.
One example for this is an ECG signal with beat annotations, where most beats are annotated as normal beats, but some beats are premature or superventricular heart beats.
The premature or superventricular heart beats can then be labelled as anomalous while the rest of the time series is normal behavior.
Overall, we collected 1354 datasets (as of May 2022).
For more details about the datasets, we refer you to the [_Datasets_ page](https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html) of the repeatability website of our evaluation paper (doi:[10.14778/3538598.3538602](https://doi.org/10.14778/3538598.3538602)).

We grouped the datasets into 24 different dataset collection for easier download and management.
The collections group datasets from a common source together, and you can download each dataset collection separately.
Each dataset is thus identified by the tuple of collection name **and** dataset name.

TimeEval uses an index-File to discover datasets.
It contains the links to the time series data and summarizes metadata about them, such as number of anomalies, contamination, input dimensionality, support for supervised or semi-supervised training of algorithms, or the time series length.
The index-File (named `datasets.csv`) for the paper's benchmark datasets can be downloaded from the repeatability page as well.

```{warning}
The _GutenTAG_ dataset collection comes with its own index-file!

The GutenTAG collection contains synthetically generated datasets using the [GutenTAG](https://github.com/HPI-Information-Systems/gutentag) dataset generator.
It is compatible to TimeEval and generates TimeEval-compatible datasets and also the necessary metadata for the index-File.
```

The downloadable ZIP-archives contain the correct folder structure, but your extraction tool might place the contained files into a new folder that is named based on the ZIP-archive-name.
The idea is that you download the index-File (`datasets.csv`) and just the dataset collections that you require, extract them all into the same folder, and then place the `datasets.csv` there.
Please note or remember the name of your datasets folder.
We will need it later and we will refer to it as `<datasets-folder>`

**Example:**

Scenario: You want to use the datasets from the CalIt2 and Daphnet collections.

Dataset download and folder structure:

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

#### Dataset download links

Please consider the repeatability page for a complete list of up-to-date download links.
This section is just for your convenience and we don't update it very frequently!

- [index-File](https://owncloud.hpi.de/s/3Cp8Q5H9gn7EVK0/download) (`datasets.csv`)
- collection 1
- collection 2

```{important}
WIP - insert download links
```

### Prepare algorithms

This section assumes that you want to use the TimeEval algorithms.
If you want to integrate your own algorithm into TimeEval, please read [](#how-to-integrate-your-own-algorithm-into-timeeval).

We collected over 70 time series anomaly detection algorithms and integrated them into TimeEval (as of May 2022).
All of our algorithm implementation make use of the {class}`~timeeval.adapters.docker.DockerAdapter` to allow you to use all features of TimeEval with them (such as resource restrictions, timeout, and fair runtime measurements).
You can find the TimeEval algorithm implementations on Github: <https://github.com/HPI-Information-Systems/TimeEval-algorithms>.
Using Docker images to bundle an algorithm for TimeEval also allows easy integration of new algorithms because there are no requirements regarding programming languages, frameworks, or tools.
Besides the many benefits, using Docker images to bundle algorithms makes preparing them for use with TimeEval a bit more cumbursome.

At the moment, we don't have the capacity to publish and maintain the algorithm's Docker images to a public Docker registry.
This means that you have to build the Docker images from scratch before you can use the algorithms with TimeEval.

```{note}
If the community demand for pre-built TimeEval algorithm images rises, we will proudly assist in publishing and mainting publicly hosted images.
However, this should be a community effort.
```

Please follow the following steps to prepare the algorithms to be evaluated with TimeEval.
For further details about the Algorithm integration concept, please read [](../concepts/algorithms.md)

0. Clone or download the [timeeval-algorithms repository](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images)
1. Build the base Docker image for your algorithm.
   You can find the image dependencies in the README-file of the repository.
   The base images are located in the folder [`0-base-images`](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).
   Please make sure that you tag your image correctly (the image name must match the `FROM`-clause in your algorithm image; **this includes the image tag**).
   To be sure, you can tag the images based on our naming scheme, which uses the prefix `registry.gitlab.hpi.de/akita/i/`.
2. Optionally build an intermediate image, such as `registry.gitlab.hpi.de/akita/i/tsmp`, required for some algorithms.
3. Build the algorithm image.

Repeat the above steps for all algorithms that you want to execute with TimeEval.
Creating a script to build all algorithm images is left as an exercise for the reader (tip: use [`find`](https://www.gnu.org/software/findutils/manual/html_node/find_html/Invoking-find.html#Invoking-find) to get the correct folder and image names, and iterate over them).
The README of the timeeval-algorithms repository contains [example calls](https://github.com/HPI-Information-Systems/TimeEval-algorithms#example-calls) to test the algorithm Docker images.

### Configure evaluation run

After we have prepared the datasets folder and the algorithm Docker images, we can install TimeEval and write an evaluation run script.
You can install TimeEval from PiPY:

```bash
pip install TimeEval
```

We recommend to create a virtual environment with conda or virtualenv for TimeEval.
The software requirements of TimeEval can found on [the home page](../index).

When TimeEval ist installed, we can use its Python API to configure an evaluation run.
We recommend to create a single Python-script for each evaluation run.
The following snippet shows the main configuration options of TimeEval:

```python
#!/usr/bin/env python3

from pathlib import Path

from timeeval import TimeEval, MultiDatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality, ResourceConstraints
from timeeval.adapters import DockerAdapter
from timeeval.params import FixedParameters
from timeeval.resource_constraints import GB


def main():
    ####################
    # Load and select datasets
    ####################
    dm = MultiDatasetManager([
        Path("<datasets-folder>")  # e.g. ./timeeval-datasets
        # you can multiple folder with an index-File to the MultiDatasetManager
    ])
    # A DatasetManager reads the index-File and allows you to access dataset metadata,
    # the datasets itself, or provides utilities to filter datasets by their metadata.
    # - select ALL available datasets
    # datasets = dm.select()
    # - select datasets from Daphnet collection
    dataset = dm.select(collection="Daphnet")
    # - select datasets with at least 2 anomalies
    # datasets = dm.select(min_anomalies=2)
    # - select multivariate datasets with a maximum contamination of 10%
    # datasets = dm.select(input_dimensionality=InputDimensionality.MULTIVARIATE, max_contamination=0.1)

    # limit to 5 datasets for this example
    datasets = datasets[:5]

    ####################
    # Load and configure algorithms
    ####################
    # create a list of algorithm-definitions, we use a single algorithm in this example
    algorithms = [Algorithm(
        name="LOF",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/lof",
            tag="latest",  # usually you can use the default here
            skip_pull=True  # set to True because the image is already present from the previous section
        ),
        # The hyperparameters of the algorithm are specified here. If you want to perform a parameter
        # search, you can also perform simple grid search with TimeEval using FullParameterGrid or
        # IndependentParameterGrid.
        param_config=FixedParameters({
            "n_neighbors": 50,
            "random_state": 42
        }),
        # required by DockerAdapter
        data_as_file=True,
        # You must specify the algorithm metadata here. The categories for all TimeEval algorithms can
        # be found in their README or their manifest.json-File.
        # UNSUPERVISED --> no training, SEMI_SUPERVISED --> training on normal data, SUPERVISED --> training on anomalies
        # if SEMI_SUPERVISED or SUPERVISED, the datasets must have a corresponding training time series
        training_type=TrainingType.UNSUPERVISED,
        # MULTIVARIATE (multi-dimensional TS) or UNIVARIATE (just a single dimension is supported)
        input_dimensionality=InputDimensionality.MULTIVARIATE
    )]

    ####################
    # Configure evaluation run
    ####################
    # set the number of repetitions of each algorithm-dataset combination (e.g. for runtime measurements):
    repetitions = 1
    # set resource constraints
    rcs = ResourceConstraints(
        task_memory_limit = 2 * GB,
        task_cpu_limit = 1.0,
    )
    
    # create TimeEval object and pass all the options
    timeeval = TimeEval(dm, datasets, algorithms,
        repetitions=repetitions,
        resource_constraints=rcs,
        # you can chose from different metrics:
        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC],
    )

    # With run(), you can start the evaluation.
    timeeval.run()
    # You can access the overall evaluation results with:
    results = timeeval.get_results()
    print(results)

    # Detailed results are automatically stored in your current working directory at ./results/<datestring>.


if __name__ == "__main__":
    main()
```

You can find more details about all exposed configuration options and methods in the [](../api).

```{important}
WIP
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
