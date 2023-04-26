from typing import Iterator


def exc_causes(exc: BaseException) -> Iterator[BaseException]:
    """Iterates over all direct causes (``__cause__``) and nested exceptions (``__context__``) of an exception.

    Parameters
    ----------
    exc : BaseException
        leave node of the exception chain

    Returns
    -------
    Iterator[BaseException]
        all linked causes and nested exceptions
    """
    yield exc
    if exc.__cause__:
        for e in exc_causes(exc.__cause__):
            yield e
    if exc.__context__:
        for e in exc_causes(exc.__context__):
            yield e
