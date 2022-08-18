# TimeEval configuration and resource restrictions

## Experiments

```{important}
WIP
```

You can configure which algorithms are executed on which datasets - to some degree.
Per default, TimeEval evaluates all algorithms on all datasets (cross product) skipping those combinations that are not compatible.
You can control which experiments are generated using the parameters `skip_invalid_combinations`, `force_training_type_match`, `force_dimensionality_match`, and `experiment_combinations_file`.
Different values for those flags allow you to achieve different goals.

### Avoiding conflicting combinations

Not all algorithms can be executed on all datasets.
If the parameter `skip_invalid_combinations` is set to `True`, TimeEval will skip all invalid combinations of algorithms and datasets based on
their input dimensionality and training type.
It is automatically enabled if either `force_training_type_match` or `force_dimensionality_match` is set to `True` (see [next section](#forcing-property-matches)).
Per default (`force_training_type_match == force_dimensionality_match == False`), the following combinations
are not executed:

- supervised algorithms on semi-supervised or unsupervised datasets (datasets cannot be used to train the algorithm)
- semi-supervised algorithm on supervised or unsupervised datasets (datasets cannot be used to train the algorithm)
- univariate algorithms on multivariate datasets (algorithm cannot process the dataset)

### Forcing property matches

force_training_type_match
Narrow down the algorithm-dataset combinations further by executing an algorithm only on datasets with **the same**
training type, e.g. unsupervised algorithms only on unsupervised datasets.
This flag implies `skip_invalid_combinations==True`.

force_dimensionality_match
Narrow down the algorithm-dataset combinations furthter by executing an algorithm only on datasets with **the same**
input dimensionality, e.g. multivariate algorithms only on multivariate datasets.
This flag implies `skip_invalid_combinations==True`.

### Selecting specific combinations

You can use the parameter `experiment_combinations_file` to supply a path to an experiment combinations CSV-File.
Using this file, you can specify explicitly which combinations of algorithms, datasets, and hyperparameters should be executed.
The file should contain CSV data with a single header line and four columns with the following names:

1. `algorithm` - name of the algorithm
2. `collection` - name of the dataset collection
3. `dataset` - name of the dataset
4. `hyper_params_id` - ID of the hyperparameter configuration

Only experiments that are present in the TimeEval configuration **and** this file are scheduled and executed.
This allows you to circumvent the cross-product that TimeEval will perform in its default configuration.

## Resource restrictions

The competitive evaluation of algorithms requires that all algorithms are executed in the same (or at least very similar) execution environment.
This means that no algorithm should have an unfair advantage over the other algorithms by having more time, memory, or other compute resource available.

TimeEval ensures comparable execution environments for all algorithms by executing algorithms in isolated [Docker](https://www.docker.com) containers and controlling their resources.
When configuring TimeEval, you can specify resource limits that apply to all evaluated algorithms in the same way.
This includes the number of CPUs, main memory, training, and execution time limits.
All those resources can be specified using a {class}`timeeval.ResourceConstraints` object.

```{important}
To make use of resource restrictions, all evaluated algorithms must be registered using the {class}`timeeval.adapters.docker.DockerAdapter`.
It is the only adapter implementation that can deal with resource constraints.
All other adapters ignore them.

TimeEval will raise an error if you try to use resource restrictions and non-`DockerAdapter`-based algorithms at the same time.
```

### Time limits

Some algorithms are not suitable for very large datasets and, thus, can take a long time until they finish either training or testing.
For this reason, TimeEval uses timeouts to restrict the runtime of all (or selected) algorithms.
You can change the timeout values for the training and testing phase globally using configuration options in
{class}`timeeval.ResourceConstraints`:

```{code-block} python
---
emphasize-lines: 9
---
from durations import Duration
from timeeval import TimeEval, ResourceConstraints

limits = ResourceConstraints(
    train_timeout=Duration("2 hours"),
    execute_timeout=Duration("2 hours"),
)
timeeval = TimeEval(dm, datasets, algorithms,
    resource_constraints=limits
)
...
```

It's also possible to use different timeouts for specific algorithms if they run using the `DockerAdapter`.
The `DockerAdapter` class can take in a `timeout` parameter that defines the maximum amount of time the algorithm is allowed to run.
The parameter takes in a {class}`durations.Duration` object as well, and overwrites the globally set timeouts.
If the timeout is exceeded, a {class}`timeeval.adapters.docker.DockerTimeoutError` is raised and the specific algorithm for the current dataset is cancelled.

### CPU and memory limits

To facilitate a fair comparison of algorithms, you can configure TimeEval to restrict the execution of algorithms to specific resources.
At the moment, TimeEval supports limiting the number of CPUs and the available memory.
GPUs are not supported by TimeEval.
The resource constraints are enforced using [explicit resource limits on the Docker containers](https://docs.docker.com/config/containers/resource_constraints/).

In the following example, we limit each algorithm to 1 CPU core and a maximum of 3 GB of memory:

```{code-block} python
from timeeval import ResourceConstraints
from timeeval.resource_constraints import GB

limits = ResourceConstraints(
    task_memory_limit = 3 * GB,
    task_cpu_limit = 1.0
)
```

If TimeEval is executed on a distributed cluster, it assumes a homogenous cluster, where all nodes of the cluster have the same capabilities and resources.
There are two options to configure resource limits for distributed TimeEval:

1. **automatically**:
   Per default, TimeEval will distribute the available resources of each node evenly to all parallel evaluation tasks.
   By changing the number of tasks per hosts, you can, thus, easily control the available resources to the evaluation tasks without worrying about over-provisioning.

   Because each tasks, in effect, trains or executes a time series anomaly detection algorithm, the tasks are resource-intensive,
   over-provisioning should be prevented.
   It could decrease overall performance and distort runtime measurements.

   However, if your compute cluster is not homogenous, TimeEval will assign different resource limits to algorithms depending on the node where the algorithm is executed on.

   **Example**:
   Non-homogenous cluster of 2 nodes with the first node {math}`A` having 10 cores and 30 GB of memory and the second node {math}`B` having 20 cores and 60 GB of memory.
   When setting the limits to `ResourceConstraints(tasks_per_hosts=10)`, algorithms executed on node {math}`A` will get 1 CPU and 3 GB of memory and algorithms executed on node {math}`B` will get 2 CPUs and 20 GB of memory.
   Therefore, always use explicit resource constraints for non-homogeneous clusters.

2. **explicitly**:
   To make sure that all algorithms have the same resource constraints and there is no over-provisioning, you should set the CPU und memory limits explicitly.
   For non-homogenous cluster take the node with the lowest overall resources and decide on how many task you want to execute in parallel.
   Then divide the available resources by the number of tasks and fix the resource limits for all algorithms to these numbers.

   **Example**:
   The same non-homogenous cluster with nodes {math}`A` and {math}`B` with 10 tasks per node (host), would result in the following constraints:

   ```{code-block} python
   rcs = ResourceConstraints(
       tasks_per_host=10,
       task_memory_limit = 3 * GB,
       task_cpu_limit = 1.0,
   )
   ```
