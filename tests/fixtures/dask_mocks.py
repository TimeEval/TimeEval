from asyncio import Future
from typing import Optional, List, Union, Generator

from timeeval.adapters.docker import DockerTimeoutError


class ExceptionForTest(Exception):
    pass


class MockDaskWorker:
    def __init__(self):
        self.address = "localhost"


class MockDaskSSHCluster:
    def __init__(self, workers: int):
        self.scheduler_address = "localhost:8000"
        self.n_workers = workers

    def close(self) -> None:
        pass

    @property
    def workers(self):
        dd = {}
        for i in range(self.n_workers):
            dd[i] = MockDaskWorker()
        return dd


class MockDaskClient:
    def __init__(self):
        self.closed = False
        self.did_shutdown = False

    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        result = task(*args, **kwargs)
        f = Future()  # type: ignore
        f.set_result(result)
        return f

    def run(self, task, *args, **kwargs):
        task(*args, **kwargs)

    def gather(self, _futures: List[Future], *args, asynchronous=False, **kwargs) -> Union[Generator[Future, None, None], bool]:
        if asynchronous:
            for _ in _futures:
                f = Future()  # type: ignore
                f.set_result(True)
                yield f
        return True

    def close(self) -> None:
        self.closed = True

    def shutdown(self) -> None:
        self.did_shutdown = True


class MockDaskExceptionClient(MockDaskClient):
    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        f = Future()  # type: ignore
        f.set_exception(ExceptionForTest("test-exception"))
        return f


class MockDaskDockerTimeoutExceptionClient(MockDaskClient):
    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        f = Future()  # type: ignore
        f.set_exception(DockerTimeoutError("test-exception-timeout"))
        return f
