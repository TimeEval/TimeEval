import json
import logging
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union, List, Generator

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss

from timeeval.datasets.metadata import DatasetId, DatasetMetadata, AnomalyLength, Stationarity, Trend, TrendType, \
    DatasetMetadataEncoder
from timeeval.utils import datasets as datasets_utils


class DatasetAnalyzer:
    def __init__(self, dataset_id: DatasetId, is_train: bool, df: Optional[pd.DataFrame] = None,
                 dataset_path: Optional[Path] = None,
                 dmgr: Optional['Datasets'] = None) -> None:  # type: ignore
        if not df and not dataset_path and not dmgr:
            raise ValueError("Either df, dataset_path, or dmgr must be supplied!")
        if not df and dmgr:
            df = dmgr.get_dataset_df(dataset_id, train=is_train)
        elif not df and dataset_path:
            df = datasets_utils.load_dataset(dataset_path)
        self.log: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._df: pd.DataFrame = df
        self.dataset_id: DatasetId = dataset_id
        self.is_train: bool = is_train
        self._log_prefix = f"[{self.dataset_id} ({'train' if self.is_train else 'test'})]"
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

    def save_to_json(self, filename: Union[str, Path], overwrite: bool = False) -> None:
        fp = Path(filename)
        metadata = []
        if fp.exists():
            if overwrite:
                self.log.warning(f"{self._log_prefix} {fp} already exists, but 'overwrite' was specified! "
                                 f"Ignoring existing contents.")
            else:
                self.log.info(f"{self._log_prefix} {fp} already exists. Reading contents and appending new metadata.")
                with open(fp, "r") as f:
                    existing_metadata = json.load(f)
                if not isinstance(existing_metadata, List) or len(existing_metadata) == 0:
                    self.log.error(f"{self._log_prefix} Existing metadata in file {fp} has the wrong format!"
                                   f"Creating backup before writing new metadata.")
                    shutil.move(fp.as_posix(), fp.parent / f"{fp.name}.bak")
                else:
                    metadata = existing_metadata
        self.log.debug(f"{self._log_prefix} Writing detailed metadata to file {filename}")
        metadata.append(self.metadata)
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, cls=DatasetMetadataEncoder)
            f.write("\n")

    @staticmethod
    def load_from_json(filename: Union[str, Path], train: bool = False) -> DatasetMetadata:
        with open(filename, "r") as f:
            metadata_list: List[DatasetMetadata] = json.load(f, object_hook=DatasetMetadataEncoder.object_hook)
            for metadata in metadata_list:
                if metadata.is_train == train:
                    return metadata
            raise ValueError(f"No metadata for {'training' if train else 'testing'} dataset in file {filename} found!")

    def _find_base_metadata(self) -> None:
        self.length = len(self._df)
        self.dimensions = len(self._df.columns) - 2
        self.contamination = len(self._df[self._df["is_anomaly"] == 1]) / self.length

        means = self._df.iloc[:, 1:-1].mean(axis=0)
        stddevs = self._df.iloc[:, 1:-1].std(axis=0)
        self.means = dict(means.items())
        self.stddevs = dict(stddevs.items())

        labels = self._df["is_anomaly"]
        label_groups = labels.groupby((labels.shift() != labels).cumsum())
        anomalies = [len(v) for k, v in label_groups if np.all(v)]
        min_anomaly_length = np.min(anomalies) if anomalies else 0
        median_anomaly_length = int(np.median(anomalies)) if anomalies else 0
        max_anomaly_length = np.max(anomalies) if anomalies else 0
        self.num_anomalies = len(anomalies)
        self.anomaly_length = AnomalyLength(
            min=min_anomaly_length,
            median=median_anomaly_length,
            max=max_anomaly_length
        )

    def _adf_stationarity_test(self, series: pd.Series, sigma: float = 0.05) -> bool:
        try:
            adftest = adfuller(series, autolag="AIC")
            adf_output = pd.Series(adftest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            self.log.debug(f"{self._log_prefix} Results of Augmented Dickey Fuller (ADF) test:\n{adf_output}")
            return adf_output["p-value"] < sigma
        except ValueError as e:
            self.log.error(f"{self._log_prefix} ADF stationarity test encountered an error: {e}")
            return False

    def _kpss_trend_stationarity_test(self, series: pd.Series, sigma: float = 0.05) -> bool:
        try:
            kpsstest = kpss(series, regression="c", nlags="auto")
            kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            self.log.debug(f"{self._log_prefix} Results of Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:\n"
                           f"{kpss_output}")
            return kpss_output["p-value"] < sigma
        except ValueError as e:
            self.log.error(f"{self._log_prefix} KPSS trend stationarity test encountered an error: {e}")
            return False

    def _analyze_series(self, series: pd.Series) -> Stationarity:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stationary = self._adf_stationarity_test(series)
            trend_stationary = self._kpss_trend_stationarity_test(series)

        if not stationary and not trend_stationary:
            stationarity = Stationarity.NOT_STATIONARY
        elif stationary and trend_stationary:
            stationarity = Stationarity.STATIONARY
        elif not stationary and trend_stationary:
            stationarity = Stationarity.TREND_STATIONARY  # detrending to make stationary
        else:  # if stationary and not trend_stationary:
            stationarity = Stationarity.DIFFERENCE_STATIONARY  # differencing to make stationary

        self.log.debug(f"{self._log_prefix} Stationarity of series '{series.name}': {stationarity}")
        return stationarity

    def _find_stationarity(self) -> None:
        """
        Idea and code adapted from:
        https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
        """
        df = self._df.set_index("timestamp").iloc[:, :-1]
        self.stationarity = {}
        for _, series in df.items():
            self.stationarity[series.name] = self._analyze_series(series)

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
        for _, series in self._df.iloc[:, 1:-1].items():
            self.trends[series.name] = [t for order in (1, 2, 3) for t in get_trend(series, order=order)]
