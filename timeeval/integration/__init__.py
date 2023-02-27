from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..timeeval import TimeEval


class TimeEvalModule(ABC):
    """Base class for TimeEval modules that add additional functionality to TimeEval.

    Inheriting classes can implement any of the following lifecycle-hooks:

    1. :meth:`~timeeval.integration.TimeEvalModule.prepare`
    2. :meth:`~timeeval.integration.TimeEvalModule.pre_run`
    3. :meth:`~timeeval.integration.TimeEvalModule.post_run`
    4. :meth:`~timeeval.integration.TimeEvalModule.finalize`

    These methods are called at the corresponding time in the TimeEval run loop. Modules can assume that the TimeEval
    configuration is already loaded and checked for user errors. If TimeEval is executed in distributed mode,
    :attr:`timeeval.TimeEval.distributed` is set to ``True`` and remoting is already set up before the first call to
    :meth:`~timeeval.integration.TimeEvalModule.prepare`.

    .. note::
        Implementing a TimeEval module is an advanced usage scenario and requires a good understanding of the internals
        of TimeEval.
    """
    def prepare(self, timeeval: TimeEval) -> None:
        """Called during the PREPARE-phase of TimeEval and before the individual algorithms' prepare-functions are
        executed.

        Parameters
        ----------
        timeeval : TimeEval
            The TimeEval instance that is currently running.
        """
        pass

    def pre_run(self, timeeval: TimeEval) -> None:
        """Called before the EVALUATION-phase of TimeEval.

        Parameters
        ----------
        timeeval : TimeEval
            The TimeEval instance that is currently running.
        """
        pass

    def post_run(self, timeeval: TimeEval) -> None:
        """Called after the EVALUATION-phase of TimeEval.

        Parameters
        ----------
        timeeval : TimeEval
            The TimeEval instance that is currently running.
        """
        pass

    def finalize(self, timeeval: TimeEval) -> None:
        """Called during the FINALIZE-phase of TimeEval and after the individual algorithms' finalize-functions were
        executed.

        Parameters
        ----------
        timeeval : TimeEval
            The TimeEval instance that is currently running.
        """
        pass
