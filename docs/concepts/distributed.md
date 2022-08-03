# Distributed TimeEval

TimeEval is able to run multiple experiments in parallel and distributedly on a cluster of multiple machines (hosts).
You can enable the distributed execution of TimeEval by setting `distributed=True` when creating the {class}`~timeeval.TimeEval` object.
The cluster configuration can be managed using a {class}`timeeval.RemoteConfiguration` object passed to the `remote_config` argument.

Distributed TimeEval will use your supplied configuration of algorithms, parameters, and datasets to create a list of experiments or evaluation tasks.
It then schedules the execution of these tasks to the nodes in the cluster so that the full cluster can be utilized for the evaluation run.
During the run, the main process monitors the execution of the tasks.
At the end of the run, it collects the evaluation results, so that you can use them for further analysis.

## Host roles

TimeEval uses {obj}`Dask's SSHCluster <dask.distributed.SSHCluster>` to dynamically create a logical cluster on a list of hosts specified by IP-addresses or hostnames.
According to Dask's terminology, TimeEval also distinguishes between the host roles _scheduler_ and _worker_.
In addition, there also is a _driver_ role.
The following table summarizes the host roles:

| Host Role   | Description                                                                                                                     |
|:------------|:--------------------------------------------------------------------------------------------------------------------------------|
| _driver_    | Host that runs the experiment script (where `python experiment-script.py` is called).                                           |
| _scheduler_ | Host that runs the {obj}`dask.Scheduler` that coordinates worker processes and distributes the tasks and jobs to the _workers_. |
| _worker_    | Host that runs one or multiple {obj}`dask.Worker`s and receives tasks and jobs. The _workers_ perform the actual computations.  |

The _driver_ host role is implicit.
The host, on which you create the {obj}`~timeeval.TimeEval`-object, gets the _driver_ role.
In the distributed mode, this main Python-process does not execute any evaluation jobs.
The _scheduler_ and _worker_ roles can be assigned to a host using the {class}`~timeeval.RemoteConfiguration` object.

The _driver_ host could be a local notebook or computer, while the _scheduler_ and _worker_ hosts are part of the cluster.
A single host can have multiple roles at the same time, and for most use cases this is totally fine.
Usually, a single machine of a cluster is used as _driver_, _scheduler_, and _worker_, while all the other machines just get the _worker_ role.
That is typically not a problem because the _driver_ and _scheduler_ components do not use many resources, and we can, therefore, use the computing power of the first host much more efficiently.

**Example**:

Assume that we have a cluster of 3 nodes (_node1_, _node2_, _node3_) and that we start a TimeEval experiment on _node1_ with the following configuration:

```python
from timeeval import RemoteConfiguration
RemoteConfiguration(
    scheduler_host="node1",
    worker_hosts=["node1", "node2", "node3"]
)
```

In this case, _node1_ executes TimeEval (role _driver_), hosts the Dask scheduler (role _scheduler_), and participates in the execution of evaluation jobs (role _worker_).
It has all three roles.
_node2_ and _node3_, however, are pure work horses and participate in the execution of evaluation jobs only (role _worker_).

## Distributed execution

If TimeEval is started with `distributed=True`, it automatically starts a Dask SSHCluster on the specified _scheduler_ and _worker_ hosts.
This is done via simple SSH-connections to the machines.
It then uses the passed experiment configurations to create evaluation jobs (called `Experiment`s).
Each `Experiment` consists of an algorithm, its hyperparameters, a dataset, and a repetition number.
After all `Experiment`s have been generated and validated, they are sent to the Dask _scheduler_ and put into a task queue.
The _workers_ pull the tasks from the _scheduler_ and perform the evaluation (i.a., executing the Docker containers of the algorithm).
All results and temporary data are stored on the disk of the local node and the overall evaluation result is sent back to the scheduler.
The _driver_ host periodically polls the _scheduler_ for the results and collects them in memory.
When all tasks have been processed, the _driver_ uses SSH connections again to pull all the temporary data from the _worker_ nodes.
This populates the local `results`-folder.

## Cluster requirements

Because we use a {obj}`Dask SSHCluster <dask.distributed.SSHCluster>` to manage the cluster hosts, there are additional requirements for every cluster node.
Please ensure that your cluster setup meets the following requirements:

- Every node must have Python and Docker installed.
- The algorithm images must be present on all nodes or Docker must be able to pull them (if `skip_pull=False`).
- Every node uses the same Python environment (the path to the Python-binary must be the same) and has TimeEval installed in it.
- The whole datasets' folder must be present on all nodes at the same path.
  This means `DatasetManager("path/to/datasets-folder", create_if_missing=False)` must work on all nodes.
- Your Python script with the experiment configuration does not import any other **local** files (e.g., `from .util import xyz`).
- All hosts must be able to reach each other via network.
- The _driver_ host must be able to open a SSH connection to all the other nodes using **passwordless SSH**.
  For this, please confirm that you can run `ssh <remote_host>` without any (password-)prompt;
  otherwise, TimeEval will not be able to reach the other nodes. (<https://www.google.com/search?q=passwordless+SSH>)
