from abc import ABC
from typing import Optional, Tuple

import numpy as np

from .metric import Metric


class RangeAucMetric(Metric, ABC):
    """Base class for range-based area under the curve metrics.

    All range-based metrics support continuous scorings and share a common implementation of the confusion matrix.
    See the subclasses' documentation for an explanation of the corresponding metric.
    """

    def __init__(self, buffer_size: Optional[int] = None, compatibility_mode: bool = False, max_samples: int = 250):
        self._buffer_size = buffer_size
        self._compat_mode = compatibility_mode
        self._max_samples = max_samples

    def anomaly_bounds(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """corresponds to range_convers_new"""
        # convert to boolean/binary
        labels = y_true > 0
        # deal with start and end of time series
        labels = np.diff(np.r_[0, labels, 0])
        # extract begin and end of anomalous regions
        index = np.arange(0, labels.shape[0])
        starts = index[labels == 1]
        ends = index[labels == -1]
        return starts, ends

    def _extend_anomaly_labels(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extends the anomaly labels with slopes on both ends. Makes the labels continuous instead of binary."""
        starts, ends = self.anomaly_bounds(y_true)

        if self._buffer_size is None:
            # per default: set buffer size as median anomaly length:
            self._buffer_size = int(np.median(ends - starts))

        if self._buffer_size <= 1:
            if self._compat_mode:
                anomalies = np.array(list(zip(starts, ends - 1)))
            else:
                anomalies = np.array(list(zip(starts, ends)))
            return y_true.astype(np.float_), anomalies

        y_true_cont = y_true.astype(np.float_)
        slope_length = self._buffer_size // 2
        length = y_true_cont.shape[0]
        if self._compat_mode:
            for i, (s, e) in enumerate(zip(starts, ends)):
                e -= 1
                x1 = np.arange(e, min(e + slope_length, length))
                y_true_cont[x1] += np.sqrt(1 - (x1 - e) / self._buffer_size)
                x2 = np.arange(max(s - slope_length, 0), s)
                y_true_cont[x2] += np.sqrt(1 - (s - x2) / self._buffer_size)
            y_true_cont = np.clip(y_true_cont, 0, 1)
            starts, ends = self.anomaly_bounds(y_true_cont)
            anomalies = np.array(list(zip(starts, ends - 1)))

        else:
            slope = np.linspace(1 / np.sqrt(2), 1, slope_length + 1)
            anomalies = np.empty((starts.shape[0], 2), dtype=np.int_)
            for i, (s, e) in enumerate(zip(starts, ends)):
                s0 = max(0, s - slope_length)
                s1 = s + 1
                y_true_cont[s0:s1] = np.maximum(slope[s0 - s1:], y_true_cont[s0:s1])
                e0 = e - 1
                e1 = min(length, e + slope_length)
                y_true_cont[e0:e1] = np.maximum(slope[e0 - e1:][::-1], y_true_cont[e0:e1])
                anomalies[i] = [s0, e1]
        return y_true_cont, anomalies

    def _uniform_threshold_sampling(self, y_score: np.ndarray) -> np.ndarray:
        if self._compat_mode:
            n_samples = 250
        else:
            n_samples = min(self._max_samples, y_score.shape[0])
        thresholds: np.ndarray = np.sort(y_score)[::-1]
        thresholds = thresholds[np.linspace(0, thresholds.shape[0] - 1, n_samples, dtype=np.int_)]
        return thresholds

    def _range_pr_roc_auc_support(self,
                                  y_true: np.ndarray,
                                  y_score: np.ndarray,
                                  with_plotting: bool = False
                                  ) -> Tuple[float, float, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        y_true_cont, anomalies = self._extend_anomaly_labels(y_true)
        thresholds = self._uniform_threshold_sampling(y_score)
        p = np.average([np.sum(y_true), np.sum(y_true_cont)])

        recalls = np.zeros(thresholds.shape[0] + 2)  # tprs
        fprs = np.zeros(thresholds.shape[0] + 2)
        precisions = np.ones(thresholds.shape[0] + 1)

        for i, t in enumerate(thresholds):
            y_pred = y_score >= t
            product = y_true_cont * y_pred
            tp = np.sum(product)
            # fp = np.dot((np.ones_like(y_pred) - y_true_cont).T, y_pred)
            fp = np.sum(y_pred) - tp
            n = len(y_pred) - p

            existence_reward = [np.sum(product[s:e + 1]) > 0 for s, e in anomalies]
            existence_reward = np.sum(existence_reward) / anomalies.shape[0]

            recall = min(tp / p, 1) * existence_reward  # = tpr
            fpr = min(fp / n, 1)
            precision = tp / np.sum(y_pred)

            recalls[i + 1] = recall
            fprs[i + 1] = fpr
            precisions[i + 1] = precision

        recalls[-1] = 1
        fprs[-1] = 1

        range_pr_auc: float = np.sum((recalls[1:-1] - recalls[:-2]) * (precisions[1:] + precisions[:-1]) / 2)
        range_roc_auc: float = np.sum((fprs[1:] - fprs[:-1]) * (recalls[1:] + recalls[:-1]) / 2)

        if with_plotting:
            return range_pr_auc, range_roc_auc, (recalls, fprs, precisions)
        return range_pr_auc, range_roc_auc, None

    def supports_continuous_scorings(self) -> bool:
        return True


class RangePrAuc(RangeAucMetric):
    def __init__(self, buffer_size: Optional[int] = None, compatibility_mode: bool = False, max_samples: int = 250,
                 plot: bool = False, plot_store: bool = False):
        super().__init__(buffer_size, compatibility_mode, max_samples)
        self._plot = plot
        self._plot_store = plot_store

    def _plot_auc(self, metric: float, plotting_details: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        import matplotlib.pyplot as plt

        tprs, fprs, precs = plotting_details
        plt.title(f"{self.name} = {metric:.2f}")
        plt.plot(tprs, precs, label=self.name, drawstyle="steps-post")
        # plt.fill_between(tprs[:-1], precs, alpha=0.1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Range recall")
        plt.ylabel("Range precision")
        if self._plot_store:
            plt.savefig(f"fig-{self.name}.pdf")
        plt.show()

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        range_pr_auc, _, plotting_details = self._range_pr_roc_auc_support(y_true, y_score, with_plotting=self._plot)
        if self._plot and plotting_details is not None:
            self._plot_auc(range_pr_auc, plotting_details)
        return range_pr_auc

    @property
    def name(self) -> str:
        return "RANGE_PR_AUC"


class RangeRocAuc(RangeAucMetric):

    def __init__(self, buffer_size: Optional[int] = None, compatibility_mode: bool = False, max_samples: int = 250,
                 plot: bool = False, plot_store: bool = False):
        super().__init__(buffer_size, compatibility_mode, max_samples)
        self._plot = plot
        self._plot_store = plot_store

    def _plot_auc(self, metric: float, plotting_details: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        import matplotlib.pyplot as plt

        tprs, fprs, precs = plotting_details
        plt.title(f"{self.name} = {metric:.2f}")
        plt.plot(fprs, tprs, label=self.name, drawstyle="steps-post")
        # plt.fill_between(fprs, tprs, alpha=0.1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("False positive rate")
        plt.ylabel("False negative rate")
        if self._plot_store:
            plt.savefig(f"fig-{self.name}.pdf")
        plt.show()

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        _, range_auc_roc, plotting_details = self._range_pr_roc_auc_support(y_true, y_score, with_plotting=self._plot)
        if self._plot and plotting_details is not None:
            self._plot_auc(range_auc_roc, plotting_details)
        return range_auc_roc

    @property
    def name(self) -> str:
        return "RANGE_ROC_AUC"


class VusMetric(Metric, ABC):
    """Base class for volume-under-surface-based metrics.

    All VUS-Metrics support continuous scorings, calculate the volume under a surface curve function, and allow plotting
    this curve function. See the subclasses' documentation for a detailed explanation of the corresponding curve and
    metric.
    """

    def __init__(self, max_buffer_size: int = 500, compatibility_mode: bool = False, max_samples: int = 250):
        super().__init__(None, compatibility_mode, max_samples)
        self._max_buffer_size = max_buffer_size

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        prs = np.zeros(self._max_buffer_size + 1)
        for bs in np.arange(0, self._max_buffer_size + 1):
            self._buffer_size = bs
            pr_auc, _, _ = self._range_pr_roc_auc_support(y_true, y_score)
            prs[bs] = pr_auc
        range_pr_volume: float = np.sum(prs) / (self._max_buffer_size + 1)
        return range_pr_volume

    @property
    def name(self) -> str:
        return "RANGE_PR_VOLUME"



class RocVUS(VusMetric):
    """Computes the volume under the receiver operating characteristic curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".

    See Also
    --------
    `https://en.wikipedia.org/wiki/Receiver_operating_characteristic <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_ : Explanation of the ROC-curve.
    """

    def __init__(self, max_buffer_size: int = 500, compatibility_mode: bool = False, max_samples: int = 250):
        super().__init__(None, compatibility_mode, max_samples)
        self._max_buffer_size = max_buffer_size

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        rocs = np.zeros(self._max_buffer_size + 1)
        for bs in np.arange(0, self._max_buffer_size + 1):
            self._buffer_size = bs
            _, roc_auc, _ = self._range_pr_roc_auc_support(y_true, y_score)
            rocs[bs] = roc_auc
        range_pr_volume: float = np.sum(rocs) / (self._max_buffer_size + 1)
        return range_pr_volume

    @property
    def name(self) -> str:
        return "RANGE_ROC_VOLUME"
