# Distributed TimeEval

TimeEval is able to run multiple tests in parallel on a cluster.
It uses [Dask's SSHCluster](https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster) to distribute tasks.
In order to use this feature, the `TimeEval` class accepts a `distributed: bool` flag
and  additional configuration options with `ssh_cluster_kwargs: dict` to setup the [SSHCluster](https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster).
