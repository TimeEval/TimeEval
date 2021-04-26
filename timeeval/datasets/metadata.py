import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Generator, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss

import timeeval.utils.datasets as datasets_utils
from timeeval.datasets.datasets import DatasetId, Datasets
from json import JSONEncoder


@dataclass
class AnomalyLength:
    min: int
    median: int
    max: int


class Stationarity(Enum):
    STATIONARY = 0
    DIFFERENCE_STATIONARY = 1
    TREND_STATIONARY = 2
    NOT_STATIONARY = 3

    @staticmethod
    def from_name(s: int) -> 'Stationarity':
        return Stationarity(s)


class TrendType(Enum):
    LINEAR = 1
    QUADRATIC = 2
    KUBIC = 3

    @staticmethod
    def from_order(order: int) -> 'TrendType':
        return TrendType(order)


@dataclass
class Trend:
    tpe: TrendType
    coef: float
    confidence_r2: float

    @property
    def name(self) -> str:
        return f"{self.tpe.name.lower()} trend"

    @property
    def order(self) -> int:
        return self.tpe.value


@dataclass
class DatasetMetadata:
    dataset_id: DatasetId
    is_train: bool
    length: int
    dimensions: int
    contamination: float
    num_anomalies: int
    anomaly_length: AnomalyLength
    means: Dict[str, float]
    stddevs: Dict[str, float]
    trends: Dict[str, List[Trend]]
    stationarities: Dict[str, Stationarity]

    @property
    def channels(self) -> int:
        return self.dimensions

    @property
    def shape(self) -> Tuple[int, int]:
        return self.length, self.dimensions

    @property
    def mean(self) -> float:
        # mypy can't infer that this actually is a float
        if not self.means:
            return 0
        return np.mean(list(self.means.values()))  # type: ignore

    @property
    def stddev(self) -> float:
        if not self.stddevs:
            return 0
        # mypy can't infer that this actually is a float
        return np.mean(list(self.stddevs.values()))  # type: ignore

    @property
    def trend(self) -> str:
        def highest_order_trend(l_trend: List[Trend]) -> Trend:
            return reduce(lambda t1, t2: t1 if t1.order >= t2.order else t2, l_trend)

        trend: Optional[Trend] = None
        for v in self.trends.values():
            if not v:
                continue
            t = highest_order_trend(v)
            if not trend:
                trend = t
            else:
                trend = highest_order_trend([t, trend])
        if trend:
            return trend.name
        else:
            return "no trend"

    @property
    def stationarity(self) -> Stationarity:
        result: Stationarity = Stationarity.STATIONARY
        for s in self.stationarities.values():
            if result and result.value < s.value:
                result = s
        return result

    def get_stationarity_name(self) -> str:
        return self.stationarity.name.lower()


class DatasetMetadataEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, DatasetMetadata) or isinstance(o, Trend) or isinstance(o, AnomalyLength):
            return asdict(o)
        elif isinstance(o, Stationarity) or isinstance(o, TrendType):
            return o.name.lower()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        else:
            return super(DatasetMetadataEncoder, self).default(o)


class DatasetAnalyzer:
    def __init__(self, dataset_id: DatasetId, is_train: bool, df: Optional[pd.DataFrame] = None,
                 dataset_path: Optional[Path] = None,
                 dmgr: Optional[Datasets] = None):
        if not df and not dataset_path and not dmgr:
            raise ValueError("Either df, dataset_path, or dmgr must be supplied!")
        if not df and dmgr:
            df = dmgr.get_dataset_df(dataset_id, train=is_train)
        elif not df and dataset_path:
            df = datasets_utils.load_dataset(dataset_path)
        self.log: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.df: pd.DataFrame = df
        self.dataset_id: DatasetId = dataset_id
        self.is_train: bool = is_train
        self._find_base_metadata()
        self._find_stationarity()
        self._find_trends()

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id=self.dataset_id,
            is_train=self.is_train,
            length=self.length,
            dimensions=self.dimensions,
            contamination=self.contamination,
            means=self.means,
            stddevs=self.stddevs,
            num_anomalies=self.num_anomalies,
            anomaly_length=self.anomaly_length,
            stationarities=self.stationarity,
            trends=self.trends,
        )

    def save_to_json(self, filename: str) -> None:
        self.log.debug(f"Writing detailed metadata about {'training' if self.is_train else 'testing'} dataset "
                       f"{self.dataset_id} to file {filename}")
        metadata = self.metadata
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, cls=DatasetMetadataEncoder)
            f.write("\n")

    def _find_base_metadata(self) -> None:
        self.length = len(self.df)
        self.dimensions = len(self.df.columns) - 2
        self.contamination = len(self.df[self.df["is_anomaly"] == 1]) / self.length

        means = self.df.iloc[:, 1:-1].mean(axis=0)
        stddevs = self.df.iloc[:, 1:-1].std(axis=0)
        self.means = dict(means.items())
        self.stddevs = dict(stddevs.items())

        labels = self.df["is_anomaly"]
        label_groups = labels.groupby((labels.shift() != labels).cumsum())
        anomalies = [len(v) for k, v in label_groups if np.all(v)]
        min_anomaly_length = np.min(anomalies)
        median_anomaly_length = int(np.median(anomalies))
        max_anomaly_length = np.max(anomalies)
        self.num_anomalies = len(anomalies)
        self.anomaly_length = AnomalyLength(
            min=min_anomaly_length,
            median=median_anomaly_length,
            max=max_anomaly_length
        )

    def _find_stationarity(self) -> None:
        """
        Idea and code adapted from:
        https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
        """
        df = self.df.set_index("timestamp").iloc[:, :-1]

        def adf_stationarity_test(series: pd.Series, sigma: float = 0.05) -> bool:
            adftest = adfuller(series, autolag="AIC")
            adf_output = pd.Series(adftest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            self.log.debug(f"Results of Augmented Dickey Fuller (ADF) test:\n{adf_output}")
            return adf_output["p-value"] < sigma

        def kpss_trend_stationarity_test(series: pd.Series, sigma: float = 0.05) -> bool:
            kpsstest = kpss(series, regression="c", nlags="auto")
            kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            self.log.debug(f"Results of Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:\n{kpss_output}")
            return kpss_output["p-value"] < sigma

        def analyze_series(series: pd.Series) -> Stationarity:
            stationary = adf_stationarity_test(series)
            trend_stationary = kpss_trend_stationarity_test(series)

            if not stationary and not trend_stationary:
                stationarity = Stationarity.NOT_STATIONARY
            elif stationary and trend_stationary:
                stationarity = Stationarity.STATIONARY
            elif not stationary and trend_stationary:
                stationarity = Stationarity.TREND_STATIONARY  # detrending to make stationary
            else:  # if stationary and not trend_stationary:
                stationarity = Stationarity.DIFFERENCE_STATIONARY  # differencing to make stationary

            self.log.debug(f"Stationarity of series '{series.name}': {stationarity}")
            return stationarity

        self.stationarity = {}
        for _, series in df.items():
            self.stationarity[series.name] = analyze_series(series)

    def _find_trends(self) -> None:
        idx = np.array(range(self.length))

        def get_trend(y: pd.Series, order: int = 1, sigma: float = 0.5) -> Generator[Trend, None, None]:
            X = np.array(np.power(idx, order)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            if r2 > sigma:
                yield Trend(
                    tpe=TrendType.from_order(order),
                    coef=model.coef_,
                    confidence_r2=r2
                )

        self.trends = {}
        for _, series in self.df.iloc[:, 1:-1].items():
            self.trends[series.name] = [t for order in (1, 2, 3) for t in get_trend(series, order=order)]
