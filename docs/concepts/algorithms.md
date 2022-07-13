# Algorithms

Any algorithm that can be called with a numpy array as parameter and a numpy array as return value can be evaluated.
TimeEval also supports passing only the filepath to an algorithm and let the algorithm perform the file reading and parsing.
In this case, the algorithm must be able to read the [TimeEval canonical file format](datasets.md#canonical-file-format).
Use `data_as_file=True` as a keyword argument to the algorithm declaration.

The `main` function of an algorithm must implement the {class}`timeeval.adapters.base.Adapter`-interface.
TimeEval comes with four different adapter types described in section [Algorithm adapters](#algorithm-adapters).

Each algorithm is associated with metadata including its learning type and input dimensionality.
TimeEval distinguishes between the three learning types {attr}`timeeval.data_types.TrainingType.UNSUPERVISED` (default),
{attr}`timeeval.data_types.TrainingType.SEMI_SUPERVISED`, and {attr}`timeeval.data_types.TrainingType.SUPERVISED`
and the two input dimensionality definitions {attr}`timeeval.data_types.InputDimensionality.UNIVARIATE` (default) and
{attr}`timeeval.data_types.InputDimensionality.MULTIVARIATE`.

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
Some of the adapters were used in earlier versions of TimeEval and are not compatible to it anymore.

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

The {class}`timeeval.adapters.distributed.JarAdapter` lets you evaluate Java algorithms in TimeEval.
You can supply the path to the Jar-File (executable) and any additional arguments to the Java-process call.

### Adapter to apply univariate methods to multivariate data

The {class}`timeeval.adapters.multivar.MultivarAdapter` allows you to apply an univariate algorithm to each dimension of a multivariate dataset individually
and receive a single aggregated result.
You can currently choose between three different result aggregation strategies that work on single points:

- {attr}`timeeval.adapters.multivar.AggregationMethod.MEAN`
- {attr}`timeeval.adapters.multivar.AggregationMethod.MEDIAN`
- {attr}`timeeval.adapters.multivar.AggregationMethod.MAX`

If `n_jobs > 1`, the algorithms are executed in parallel.
