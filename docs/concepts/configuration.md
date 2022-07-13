# TimeEval configuration and resource restrictions

tbd

## Using time limits

Some algorithms are not suitable for very large datasets and, thus, can take a long time until they finish either training or testing.
For this reason, TimeEval uses timeouts to restrict the runtime of all (or selected) algorithms.
You can change the timeout values for the training and testing phase globally using configuration options in
{class}`timeeval.resource_constraints.ResourceConstraints`:

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

```{important}
Currently, only {class}`timeeval.adapters.docker.DockerAdapter` can deal with resource constraints.
All other adapters ignore them.
```

It's also possible to use different timeouts for specific algorithms if they run using the `DockerAdapter`.
The `DockerAdapter` class can take in a `timeout` parameter that defines the maximum amount of time the algorithm is allowed to run.
The parameter takes in a {class}`durations.Duration` object as well, and overwrites the globally set timeouts.
If the timeout is exceeded, a {class}`timeeval.adapters.docker.DockerTimeoutError` is raised and the specific algorithm for the current dataset is cancelled.
