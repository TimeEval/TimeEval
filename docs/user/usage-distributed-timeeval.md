# (Advanced) Distributed execution of TimeEval

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

Similar to the [local execution of TimeEval](usage-timeeval.md), we also have to prepare the datasets and algorithms first.

## Prepare time series datasets

1. Please create a dataset folder on each node using the same path. For example: `/data/timeeval-datasets`.
2. Copy your datasets and also the index-file (`datasets.csv`) to all nodes.
3. Test if TimeEval can access this folder and find your datasets on each node:

   ```python
   from timeeval import DatasetManager

   dmgr = DatasetManager("/data/timeeval-datasets", create_if_missing=False)
   dataset = dmgr.get(("<your-collection-name>", "<your-dataset-name>"))
   ```

## Prepare algorithms

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

## TimeEval configuration for distributed execution

Setting up TimeEval for distributed execution follows the same principles as for [local execution](usage-timeeval.md#configure-evaluation-run).
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
