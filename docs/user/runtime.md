# Repetitive runs and measuring runtime

TimeEval has the ability to run an experiment multiple times to improve runtime measurements.
Therefore, {class}`timeeval.TimeEval` has the parameter `repetitions: int = 1`, which tells TimeEval how many times to execute each experiment (algorithm, hyperparameters, and dataset combination).

When measuring runtime, we highly recommend to use TimeEval's feature to limit each algorithm to a specific set of resources (meaning CPU and memory).
This requires using the {class}`timeeval.adapters.docker.DockerAdapter` for the algorithms.
See the concept page [](../concepts/configuration.md) for more details.

To retrieve the aggregated results, you can call {func}`timeeval.TimeEval.get_results` with the parameter `aggregated: bool = True`.
This aggregates the quality and runtime measurements  using mean and standard deviation.
Erroneous experiments are excluded from an aggregate.
For example, if you have `repetitions = 5` and one of five experiments failed, the average is built only over the 4 successful runs.
