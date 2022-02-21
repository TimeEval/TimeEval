from dataclasses import dataclass
from typing import Optional, Tuple

import psutil
from durations import Duration

MB = 1024 ** 2
GB = 1024 ** 3
DEFAULT_TASKS_PER_HOST = 1
DEFAULT_TIMEOUT = Duration("8 hours")


@dataclass
class ResourceConstraints:
    """
    **Resource constraints are not supported by all algorithm Adapters!**

    For docker: Swap is always disabled. Resource constraints are enforced using explicit resource limits on the Docker
    container.
    """

    tasks_per_host: int = DEFAULT_TASKS_PER_HOST
    task_memory_limit: Optional[int] = None
    task_cpu_limit: Optional[float] = None
    train_timeout: Duration = DEFAULT_TIMEOUT
    execute_timeout: Duration = DEFAULT_TIMEOUT
    use_preliminary_model_on_train_timeout: bool = True
    use_preliminary_scores_on_execute_timeout: bool = True

    def get_compute_resource_limits(self,
                                    memory_overwrite: Optional[int] = None,
                                    cpu_overwrite: Optional[float] = None) -> Tuple[int, float]:
        """
        Calculates the resource constraints for a single task. **Must be called on the node that will execute the
        task!**

        There are three sources for resource limits (in decreasing priority):

        1. Overwrites (passed to this function as arguments)
        2. Explicitly set resource limits (on this object using `task_memory_limit` and `task_cpu_limit`)
        3. Default resource constraints

        **Overall default**:

        1 task per node using all available cores and RAM (except small margin for OS).

        When multiple tasks are specified, the resources are equally shared between all concurrent tasks. This means
        that CPU limit is set based on node CPU count divided by the number of tasks and memory limit is set based on
        total memory of node minus 1 GB (for OS) divided by the number of tasks.

        :param memory_overwrite: if this is set, it will overwrite the memory limit
        :param cpu_overwrite: if this is set, it will overwrite the CPU limit
        :return: Tuple of memory and CPU limit.
            Memory limit is expressed in Bytes and CPU limit is expressed in fractions of CPUs (e.g. 0.25 means: only
            use 1/4 of a single CPU core).
        """
        if memory_overwrite:
            memory_limit = memory_overwrite
        elif self.task_memory_limit:
            memory_limit = self.task_memory_limit
        else:
            usable_memory = psutil.virtual_memory().total - 1 * GB
            memory_limit = usable_memory // self.tasks_per_host

        if cpu_overwrite:
            cpu_limit = cpu_overwrite
        elif self.task_cpu_limit:
            cpu_limit = self.task_cpu_limit
        else:
            cpus = psutil.cpu_count()
            cpu_limit = cpus / self.tasks_per_host

        return memory_limit, cpu_limit

    def get_train_timeout(self, timeout_overwrite: Optional[Duration] = None) -> Duration:
        """Returns the maximum runtime of a training task in seconds.

        :param timeout_overwrite: if this is set, it will overwrite the global timeout
        """
        return self._get_timeout_with_overwrite(self.train_timeout, timeout_overwrite)

    def get_execute_timeout(self, timeout_overwrite: Optional[Duration] = None) -> Duration:
        """Returns the maximum runtime of an execution task in seconds.

        :param timeout_overwrite: if this is set, it will overwrite the global timeout
        """
        return self._get_timeout_with_overwrite(self.execute_timeout, timeout_overwrite)

    @staticmethod
    def _get_timeout_with_overwrite(timeout: Duration, overwrite: Optional[Duration]) -> Duration:
        if overwrite is not None:
            timeout = overwrite
        return timeout

    @staticmethod
    def default_constraints() -> 'ResourceConstraints':
        return ResourceConstraints()
