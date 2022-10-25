# Using TimeEval to evaluate algorithms

TimeEval is an evaluation tool for time series anomaly detection algorithms.
We provide a large collection of compatible datasets and algorithms.
The following section describes how you can set up TimeEval to perform your own experiments using the provided datasets and algorithms.
The process consists of three steps: [preparing the datasets](#prepare-datasets), [preparing the algorithms](#prepare-algorithms), and writing the [experiment script](#configure-evaluation-run).

## Prepare datasets

This section assumes that you want to use the TimeEval datasets.
If you want to use your own datasets with TimeEval, please read [](integrate-dataset.md).

For the evaluation of time series anomaly detection algorithms, we collected univariate and multivariate time series datasets from various sources.
We looked out for real-world as well as synthetically generated datasets with real-valued values and anomaly annotations.
We included datasets with direct anomaly annotations (points or subsequences are labelled as either normal (0) or anomalous (1)) and indirect anomaly annotations.
For the latter, we included datasets with categorical labels, where a single class (or low number of classes) is clearly underrepresented and can be assigned to unwanted, erroneous, or anomalous behavior.
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
We will need it later, and we will refer to it as `<datasets-folder>`

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

### Dataset download links

Please consider the repeatability page for a complete list of up-to-date download links.
This section is just for your convenience, and we don't update it very frequently!

- [index-File ⬇](https://nextcloud.hpi.de/s/3ciMX4yAn8yC5Lb/download) (`datasets.csv`, <1MB)
- [CalIt2 ⬇](https://nextcloud.hpi.de/s/i4HNX5StwBdkjFJ/download) (<1MB)
- [Daphnet ⬇](https://nextcloud.hpi.de/s/sD8KNDF7JBR3Ajo/download) (15MB)
- [Dodgers ⬇](https://nextcloud.hpi.de/s/kj8fi8csPdsRXws/download) (<1MB))
- [Exathlon ⬇](https://nextcloud.hpi.de/s/ME9EJcpBa5i4HGq/download) (106MB))
- [GHL ⬇](https://nextcloud.hpi.de/s/trKcAzSxm3A4PMW/download) (153MB)
- [Genesis ⬇](https://nextcloud.hpi.de/s/Y3yq3CnnMarXzSJ/download) (<1MB)
- [KDD-TSAD ⬇](https://nextcloud.hpi.de/s/3ZbWw478teRJLzB/download) (110MB)
- [Kitsune ⬇](https://nextcloud.hpi.de/s/gLPAPbwj2TgAi3j/download) (13.5GB)
- [LTDB ⬇](https://nextcloud.hpi.de/s/z8Nt5NgBfDnJbzY/download) (405MB)
- [MGAB ⬇](https://nextcloud.hpi.de/s/4ByzxjWmAALb5Tn/download) (12MB)
- [MITDB ⬇](https://nextcloud.hpi.de/s/8yoLBDC5ezMwe9R/download) (176MB)
- [Metro ⬇](https://nextcloud.hpi.de/s/wSCxtM4Y6PHMmD7/download) (<1MB)
- [NAB ⬇](https://nextcloud.hpi.de/s/bec4p8XNEGNWTwP/download) (2MB)
- [NASA-MSL ⬇](https://nextcloud.hpi.de/s/w9332jso24yHijZ/download) (<1MB)
- [NASA-SMAP ⬇](https://nextcloud.hpi.de/s/CiGza9EQ5fxRS9F/download) (2MB)
- [NormA ⬇](https://nextcloud.hpi.de/s/5sC7Pb2PowdZFFK/download) (15MB)
- [OPPORTUNITY ⬇](https://nextcloud.hpi.de/s/nYtZ5mTLpYX2G7p/download) (204MB)
- [Occupancy ⬇](https://nextcloud.hpi.de/s/6bxznJmk3PHbz7Z/download) (<1MB)
- [SMD ⬇](https://nextcloud.hpi.de/s/LSfo8wLW77yrZEY/download) (99MB)
- [SVDB ⬇](https://nextcloud.hpi.de/s/ScHgYbP7eD8Dtnq/download) (103MB)
- [TSB-UAD synthetic ⬇](https://nextcloud.hpi.de/s/dY2K6ZG9QLkqrZj/download) (1.8GB)
- [TSB-UAD artificial ⬇](https://nextcloud.hpi.de/s/FyGRCMXKWqH7kHw/download) (178MB)
- [GutenTAG ⬇](https://nextcloud.hpi.de/s/5fPgiDQW5iLbwi3/download) (own `datasets.csv`-File, 106MB)

## Prepare algorithms

This section assumes that you want to use the TimeEval algorithms.
If you want to integrate your own algorithm into TimeEval, please read [](integrate-algorithm.md).

We collected over 70 time series anomaly detection algorithms and integrated them into TimeEval (as of May 2022).
All of our algorithm implementation make use of the {class}`~timeeval.adapters.docker.DockerAdapter` to allow you to use all features of TimeEval with them (such as resource restrictions, timeout, and fair runtime measurements).
You can find the TimeEval algorithm implementations on GitHub: <https://github.com/HPI-Information-Systems/TimeEval-algorithms>.
Using Docker images to bundle an algorithm for TimeEval also allows easy integration of new algorithms because there are no requirements regarding programming languages, frameworks, or tools.
Besides the many benefits, using Docker images to bundle algorithms makes preparing them for use with TimeEval a bit more cumbursome.

At the moment, we don't have the capacity to publish and maintain the algorithm's Docker images to a public Docker registry.
This means that you have to build the Docker images from scratch before you can use the algorithms with TimeEval.

```{note}
If the community demand for pre-built TimeEval algorithm images rises, we will proudly assist in publishing and mainting publicly hosted images.
However, this should be a community effort.
```

Please follow the following steps to prepare the algorithms to be evaluated with TimeEval.
For further details about the Algorithm integration concept, please read the concept page [](../concepts/algorithms.md).

1. Clone or download the [timeeval-algorithms repository](https://github.com/HPI-Information-Systems/TimeEval-algorithms)
2. Build the base Docker image for your algorithm.
   You can find the image dependencies in the README-file of the repository.
   The base images are located in the folder [`0-base-images`](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).
   Please make sure that you tag your image correctly (the image name must match the `FROM`-clause in your algorithm image; **this includes the image tag**).
   To be sure, you can tag the images based on our naming scheme, which uses the prefix `registry.gitlab.hpi.de/akita/i/`.
3. Optionally build an intermediate image, such as `registry.gitlab.hpi.de/akita/i/tsmp`, required for some algorithms.
4. Build the algorithm image.

Repeat the above steps for all algorithms that you want to execute with TimeEval.
Creating a script to build all algorithm images is left as an exercise for the reader (tip: use [`find`](https://www.gnu.org/software/findutils/manual/html_node/find_html/Invoking-find.html#Invoking-find) to get the correct folder and image names, and iterate over them).
The README of the timeeval-algorithms repository contains [example calls](https://github.com/HPI-Information-Systems/TimeEval-algorithms#example-calls) to test the algorithm Docker images.

## Configure evaluation run

After we have prepared the datasets folder and the algorithm Docker images, we can install TimeEval and write an evaluation run script.
You can install TimeEval from [PyPI](https://pypi.org/project/TimeEval):

```bash
pip install TimeEval
```

We recommend to create a virtual environment with conda or virtualenv for TimeEval.
The software requirements of TimeEval can be found on [the home page](../index).

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
        # you can add multiple folders with an index-File to the MultiDatasetManager
    ])
    # A DatasetManager reads the index-File and allows you to access dataset metadata,
    # the datasets itself, or provides utilities to filter datasets by their metadata.
    # - select ALL available datasets
    # datasets = dm.select()
    # - select datasets from Daphnet collection
    datasets = dm.select(collection="Daphnet")
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
        # you can choose from different metrics:
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

You can find more details about all exposed configuration options and methods in the [](../api/index.rst).

If you are able to successfully execute the previous example evaluation, you can find more information at the following locations:

- [Add your own algorithm to TimeEval](integrate-algorithm.md)
- [Add your own datasets to TimeEval](integrate-dataset.md)
- [Configure TimeEval and resource constraints](../concepts/configuration.md)
- [Configure algorithm hyperparameters](../concepts/params.md)
- [Using custom metrics](custom-metrics.md)
- [Measuring algorithm runtime](runtime.md)
- [Executing TimeEval distributedly](usage-distributed-timeeval.md)
