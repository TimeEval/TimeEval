# Algorithms

Any algorithm that can be called with a numpy array as parameter and a numpy array as return value can be evaluated.
TimeEval also supports passing only the filepath to an algorithm and let the algorithm perform the file reading and parsing.
In this case, the algorithm must be able to read the [TimeEval canonical file format](datasets.md#canonical-file-format).
Use `data_as_file=True` as a keyword argument to the algorithm declaration.

The `main` function of an algorithm must implement the {class}`timeeval.adapters.base.Adapter`-interface.
TimeEval comes with four different adapter types described in section [Algorithm adapters](#algorithm-adapters).

Each algorithm is associated with metadata including its learning type and input dimensionality.
TimeEval distinguishes between the three learning types {attr}`timeeval.TrainingType.UNSUPERVISED` (default),
{attr}`timeeval.TrainingType.SEMI_SUPERVISED`, and {attr}`timeeval.TrainingType.SUPERVISED`
and the two input dimensionality definitions {attr}`timeeval.InputDimensionality.UNIVARIATE` (default) and
{attr}`timeeval.InputDimensionality.MULTIVARIATE`.

## Registering algorithms

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

timeeval = TimeEval(DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK]), datasets, algorithms)
```

## Algorithm adapters

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

### Function adapter

The {class}`timeeval.adapters.function.FunctionAdapter` allows you to use Python functions and methods as the algorithm
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

### Docker adapter

The {class}`timeeval.adapters.docker.DockerAdapter` allows you to run an algorithm as a Docker container.
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

```{important}
Using a `DockerAdapter` implies that `data_as_file=True` in the `Algorithm` construction.
The adapter supplies the dataset to the algorithm via bind-mounting and does not support passing the data as numpy array.
```

## Experimental algorithm adapters

The algorithm adapters in this section are prototypical implementations and not fully tested with TimeEval.
Some adapters were used in earlier versions of TimeEval and are not compatible to it anymore.

```{warning}
The following algorithm adapters should be used for educational purposes only.
They are not fully tested with TimeEval!
```

### Distributed adapter

The {class}`timeeval.adapters.distributed.DistributedAdapter` allows you to execute an already distributed algorithm on multiple machines.
Supply a list of `remote_hosts` and a `remote_command` to this adapter.
It will use SSH to connect to the remote hosts and execute the `remote_command` on these hosts before starting the main algorithm locally.

```{important}
- Password-less ssh to the remote machines required!
- **Do not combine with the distributed execution of TimeEval ("TimeEval.Distributed" using `TimeEval(..., distributed=True)`)!**
  This will affect the timing results.
```

### Jar adapter

The {class}`timeeval.adapters.jar.JarAdapter` lets you evaluate Java algorithms in TimeEval.
You can supply the path to the Jar-File (executable) and any additional arguments to the Java-process call.

### Adapter to apply univariate methods to multivariate data

The {class}`timeeval.adapters.multivar.MultivarAdapter` allows you to apply an univariate algorithm to each dimension of a multivariate dataset individually
and receive a single aggregated result.
You can currently choose between three different result aggregation strategies that work on single points:

- {attr}`timeeval.adapters.multivar.AggregationMethod.MEAN`
- {attr}`timeeval.adapters.multivar.AggregationMethod.MEDIAN`
- {attr}`timeeval.adapters.multivar.AggregationMethod.MAX`

If `n_jobs > 1`, the algorithms are executed in parallel.

## Algorithms provided with TimeEval

All algorithms that we provide with TimeEval use the {class}`~timeeval.adapters.docker.DockerAdapter` as adapter-implementation to allow you to use all features of TimeEval with them (such as resource restrictions, timeout, and fair runtime measurements).
You can find the TimeEval algorithm implementations on Github: <https://github.com/HPI-Information-Systems/TimeEval-algorithms>.
Using Docker images to bundle an algorithm for TimeEval also allows easy integration of new algorithms because there are no requirements regarding programming languages, frameworks, or tools.
However, using Docker images to bundle algorithms makes preparing them for use with TimeEval a bit more cumbursome (cf. [](../user/integrate-algorithm.md)).

In this section, we describe some important aspects of this architecture.

### TimeEval base Docker images

To benefit from Docker layer caching and to reduce code duplication (DRY!), we decided to put common functionality in so-called base images.
The following is taken care of by base images:

- Provide system (OS and common OS tools)
- Provide language runtime (e.g. python3, java8)
- Provide common libraries / algorithm dependencies
- Define volumes for IO
- Define Docker entrypoint script (performs initial container setup before the algorithm is executed)

Currently, we provide the following root base images:

| Name/Folder | Image | Usage |
| :--- | :---- | :---- |
| python2-base | `registry.gitlab.hpi.de/akita/i/python2-base` | Base image for TimeEval methods that use python2 (version 2.7); includes default python packages. |
| python3-base | `registry.gitlab.hpi.de/akita/i/python3-base` | Base image for TimeEval methods that use python3 (version 3.7.9); includes default python packages. |
| python36-base | `registry.gitlab.hpi.de/akita/i/python36-base` | Base image for TimeEval methods that use python3.6 (version 3.6.13); includes default python packages. |
| r-base | `registry.gitlab.hpi.de/akita/i/r-base` | Base image for TimeEval methods that use R (version 3.5.2-1). |
| r4-base | `registry.gitlab.hpi.de/akita/i/r4-base` | Base image for TimeEval methods that use R (version 4.0.5). |
| java-base | `registry.gitlab.hpi.de/akita/i/java-base` | Base image for TimeEval methods that use Java (JRE 11.0.10). |

In addition to the root base images, we also provide some derived base images that add further common functionality to the language runtimes:

| Name/Folder | Image | Usage |
| :--- | :---- | :---- |
| tsmp | `registry.gitlab.hpi.de/akita/i/tsmp` | Base image for TimeEval methods that use the matrix profile R package [`tsmp`](https://github.com/matrix-profile-foundation/tsmp); is based on `registry.gitlab.hpi.de/akita/i/r-base`. |
| pyod | `registry.gitlab.hpi.de/akita/i/pyod` | Base image for TimeEval methods that are based on the [`pyod`](https://github.com/yzhao062/pyod) library; is based on `registry.gitlab.hpi.de/akita/i/python3-base` |
| timeeval-test-algorithm | `registry.gitlab.hpi.de/akita/i/timeeval-test-algorithm` | Test image for TimeEval tests that use docker; is based on `registry.gitlab.hpi.de/akita/i/python3-base`. |
| python3-torch | `registry.gitlab.hpi.de/akita/i/python3-torch` | Base image for TimeEval methods that use python3 (version 3.7.9) and PyTorch (version 1.7.1); includes default python packages and torch; is based on `registry.gitlab.hpi.de/akita/i/python3-base`. |

You can find all current base images in the [`timeeval-algorithms`-repository](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).

### TimeEval algorithm interface

TimeEval uses a common interface to execute all the algorithms that implement the {class}`~timeeval.adapters.docker.DockerAdapter`.
This means that the algorithms' input, output, and parameterization is equal for all provided algorithms.

#### Execution and parametrization

All algorithms are executed by creating a Docker container using their Docker image and then executing it.
The base images take care of the container startup and they call the main algorithm file with a single positional parameter.
This parameter contains a String-representation of the algorithm configuration as JSON.
Example parameter JSON (2022-08-18):

```python
{
  "executionType": 'train' | 'execute',
  "dataInput": string,   # example: "path/to/dataset.csv",
  "dataOutput": string,  # example: "path/to/results.csv",
  "modelInput": string,  # example: "/path/to/model.pkl",
  "modelOutput": string, # example: "/path/to/model.pkl",
  "customParameters": dict
}
```

#### Custom algorithm parameters

All algorithm hyper parameters described in the correspoding algorithm paper are exposed via the `customParameters` configuration option.
This allows us to set those parameters from TimeEval.

```{warning}
TimeEval does **not** parse a `manifest.json` file to get the custom parameters' types and default values.
We expect the users of TimeEval to be familiar with the algorithms, so that they can specify the required parameters manually.
However, we require each algorithm to be executable without specifying any custom parameters (especially for testing purposes).
Therefore, **please provide sensible default parameters for all custom parameters within the method's code**.

If you want to contribute your algorithm implementation to TimeEval, please add a `manifest.json`-file to your algorithm anyway to aid the integration into other tools and for user information.

If your algorithm does not use the default parameters automatically and expects them to be provided, your algorithm will fail during runtime if no parameters are provided by the TimeEval user.
```

#### Input and output

Input and output for an algorithm is handled via bind-mounting files and folders into the Docker container.

All **input data**, such as the training dataset and the test dataset, are mounted read-only to the `/data`-folder of the container.
The configuration options `dataInput` and `modelInput` reflect this with the correct path to the dataset (e.g. `{ "dataInput": "/data/dataset.test.csv" }`).
The dataset format follows our [](./datasets.md#canonical-file-format).

All **output** of your algorithm should be written to the `/results`-folder.
This is also reflected in the configuration options with the correct paths for `dataOutput` and `modelOutput` (e.g. `{ "dataOutput": "/results/anomaly_scores.csv" }`).
The `/results`-folder is also bind-mounted to the algorithm container - but writable -, so that TimeEval can access the results after your algorithm finished.
An algorithm can also use this folder to write persistent log and debug information.

Every algorithm must produce an **anomaly scoring** as output and put it at the location specified with the `dataOutput`-key in the configuration.
The output file's format is CSV-based with a single column and no header.
You can for example produce a correct anomaly scoring with NumPy's {obj}`numpy.savetxt`-function: `np.savetxt(<args.dataOutput>, arr, delimiter=",")`.

**Temporary files** and data of an algorithm are written to the current working directory (currently this is `/app`) or the temporary directory `/tmp` within the Docker container.
All files written to those folders is lost after the algorithm container is removed.

#### Example calls

The following Docker command represents the way how the TimeEval {class}`~timeeval.adapters.docker.DockerAdapter` executes your algorithm image:

```bash
docker run --rm \
    -v <path/to/dataset.csv>:/data/dataset.csv:ro \
    -v <path/to/results-folder>:/results:rw \
    -e LOCAL_UID=<current user id> \
    -e LOCAL_GID=<groupid of akita group> \
    <resource restrictions> \
  registry.gitlab.hpi.de/akita/i/<your_algorithm>:latest execute-algorithm '{ \
    "executionType": "execute", \
    "dataInput": "/data/dataset.csv", \
    "modelInput": "/results/model.pkl", \
    "dataOutput": "/results/anomaly_scores.ts", \
    "modelOutput": "/results/model.pkl", \
    "customParameters": {} \
  }'
```

This is translated to the following call within the container from the entry script of the base image:

```bash
docker run --rm \
    -v <path/to/dataset.csv>:/data/dataset.csv:ro \
    -v <path/to/results-folder>:/results:rw <...>\
  registry.gitlab.hpi.de/akita/i/<your_algorithm>:latest bash
# now, within the container
<python | java -jar | Rscript> $ALGORITHM_MAIN '{ \
  "executionType": "execute", \
  "dataInput": "/data/dataset.csv", \
  "modelInput": "/results/model.pkl", \
  "dataOutput": "/results/anomaly_scores.ts", \
  "modelOutput": "/results/model.pkl", \
  "customParameters": {} \
}'
```
