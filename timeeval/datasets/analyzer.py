import json
import logging
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union, List, Generator, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, kpss

from .metadata import DatasetId, DatasetMetadata, AnomalyLength, Stationarity, Trend, TrendType, DatasetMetadataEncoder
from ..utils import datasets as datasets_utils


class DatasetAnalyzer:
    """Utility class to analyze a dataset and infer metadata about the dataset.

    Use this class to compute necessary metadata from a time series. The computation is started directly when
    instantiating this class. You can access the results using the property ``metadata``. There multiple ways to
    instantiate this class, but you always have to specify the dataset ID, because it is part of the metadata:

        1. Use an existing pandas data frame object. Supply a value to the parameter `df`.
        2. Use a path to a time series. Supply a value to the parameter `dataset_path`.
        3. Use a dataset ID and a reference to the dataset manager. Supply a value to the parameter `dmgr`.

    This class computes simple metadata, such as number of anomalies, mean, and standard deviation, as well as advanced
    metadata, such as trends or stationarity information for all time series channels. The simple metadata is exact. But
    the advanced metadata is estimated based on the observed time series data. The trend is computed by fitting linear
    regression models of different order to the time series. If the regression has a high enough correlation with the
    observed values, the trends and their confidence are recorded. The stationarity of the time series is estimated
    using two statistical tests, the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) and the Augmented Dickey Fuller (ADF)
    test.

    The metadata of a dataset can be stored to disk. This class provides utility functions to create a JSON-file per
    dataset, containing the metadata about the test time series and the optional training time series.

    Parameters
    ----------
    dataset_id : tuple of str, str
        ID of the dataset consisting of collection and dataset name.
    is_train : bool
        If the analyzed time series is the testing or training time series of the dataset.
    df : data frame, optional
        Time series data frame. If `df` is supplied, you can omit `dataset_path` and `dmgr`.
    dataset_path : path, optional
        Path to the time series. If `dataset_path` is supplied, you can omit `df` and `dmgr`.
    dmgr : Datasets object, optional
        Dataset manager instance that is used to load the time series if `df` and `dataset_path` are not specified.
    ignore_stationarity : bool, optional
        Don't estimate the time series' channels stationarity. This might be necessary for large datasets, because this
        step takes a lot of time.
    ignore_trend : bool, optional
        Don't estimate the time series' channels trend type. This might be necessary for large datasets, because this
        step takes a lot of time.

    See Also
    --------
    :class:`statsmodels.tsa.stattools.adfuller`
    :class:`statsmodels.tsa.stattools.kpss`
    """

    def __init__(self, dataset_id: DatasetId, is_train: bool,
                 df: Optional[pd.DataFrame] = None,
                 dataset_path: Optional[Path] = None,
                 dmgr: Optional['Datasets'] = None,  # type: ignore
                 ignore_stationarity: bool = False,
                 ignore_trend: bool = False) -> None:
        if df is None and not dataset_path and dmgr is None:
            raise ValueError("Either df, dataset_path, or dmgr must be supplied!")
        if df is None and dmgr:
            df = dmgr.get_dataset_df(dataset_id, train=is_train)
        elif df is None and dataset_path:
            df = datasets_utils.load_dataset(dataset_path)
        self.log: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._df: pd.DataFrame = df
        self.dataset_id: DatasetId = dataset_id
        self.is_train: bool = is_train
        self._log_prefix = f"[{self.dataset_id} ({'train' if self.is_train else 'test'})]"
        self._find_base_metadata()
        if ignore_stationarity:
            self.stationarity = dict(
                (str(series.name), Stationarity.NOT_STATIONARY)
                for _, series in self._df.iloc[:, 1:-1].items()
            )
        else:
            self._find_stationarity()
        if ignore_trend:
            self.trends: Dict[str, List[Trend]] = {}
        else:
            self._find_trends()

    @property
    def metadata(self) -> DatasetMetadata:
        """Returns the computed metadata about the time series."""
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
        """Save the computed metadata for a dataset to disk.

        This method writes a dataset's metadata to a JSON-formatted file to disk. The file contains a list of
        metadata specifications. One specification for the test time series and potentially another one for the test
        time series. Since the DatasetAnalyzer just analyzes a single time series at a time, this method appends the
        current metadata to the existing list per default. If you want to overwrite the existing content of the file,
        you can use the parameter `overwrite`.

        Parameters
        ----------
        filename: path
            Path to the file, where the metadata should be written to. Might already exist.
        overwrite: bool
            If existing data in the file should be overwritten or the current metadata should just be added to it.
        """
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
        """Loads existing time series metadata from disk.

        If there are multiple metadata entries with the same dataset ID and training/testing-label, the first entry is
        used.

        Parameters
        ----------
        filename: path
            Path to the JSON-file containing the dataset metadata. Can be written using
            :func:`timeeval.datasets.DatasetAnalyzer.save_to_json`.
        train: bool
            Whether the training or testing time series' metadata should be loaded from the file.

        Returns
        -------
        metadata: time series metadata object
            Metadata of the training or testing time series.
        """
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
        except Exception as e:
            self.log.error(f"{self._log_prefix} ADF stationarity test for {series.name} encountered an error: {e}")
            return False

    def _kpss_trend_stationarity_test(self, series: pd.Series, sigma: float = 0.05) -> bool:
        try:
            kpsstest = kpss(series, regression="c", nlags="auto")
            kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            self.log.debug(f"{self._log_prefix} Results of Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:\n"
                           f"{kpss_output}")
            return kpss_output["p-value"] < sigma
        except Exception as e:
            self.log.error(
                f"{self._log_prefix} KPSS trend stationarity test for {series.name} encountered an error: {e}")
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
        df = self._df
        # Circumvent missing "timestamp" header:
        df.columns = ["timestamp" if "Unnamed" in c else c for c in df.columns]
        df = df.set_index("timestamp").iloc[:, :-1]
        self.stationarity = {}
        for _, series in df.items():
            self.stationarity[series.name] = self._analyze_series(series)

    def _find_trends(self) -> None:
        idx = np.array(range(self.length))

        def get_trend(y: pd.Series, order: int = 1, sigma: float = 0.5) -> Generator[Trend, None, None]:
            try:
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
            except Exception as e:
                self.log.error(
                    f"{self._log_prefix} trend analysis for {y.name} encountered an error: {e}")

        self.trends = {}
        for _, series in self._df.iloc[:, 1:-1].items():
            self.trends[series.name] = [t for order in (1, 2, 3) for t in get_trend(series, order=order)]
