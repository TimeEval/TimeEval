import json
from copy import deepcopy
from dataclasses import dataclass, asdict
from enum import Enum
from functools import reduce
from json import JSONEncoder
from typing import Tuple, List, Optional, Dict, Any

import numpy as np


DatasetId = Tuple[str, str]


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
    """Represents the metadata of a single time series of a dataset (for each channel)."""
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
        if not self.means:
            return 0
        # mypy can't infer that this actually is a float
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

    def to_json(self, pretty: bool = False) -> str:
        return json.dumps(self, cls=DatasetMetadataEncoder,
                          indent=2 if pretty else None,
                          sort_keys=True if pretty else False)

    @staticmethod
    def from_json(s: str) -> 'DatasetMetadata':
        return json.loads(s, object_hook=DatasetMetadataEncoder.object_hook)


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

    @staticmethod
    def object_hook(dct: dict) -> Any:
        if "anomaly_length" in dct and "dataset_id" in dct and "is_train" in dct and "contamination" in dct:
            anomaly_length_dict = dct["anomaly_length"]
            anomaly_length = AnomalyLength(
                min=anomaly_length_dict["min"],
                median=anomaly_length_dict["median"],
                max=anomaly_length_dict["max"]
            )
            trends = deepcopy(dct["trends"])
            for k, obj in trends.items():
                trends[k] = [Trend(TrendType[t["tpe"].upper()], t["coef"], t["confidence_r2"]) for t in obj]
            stationarities_dict = deepcopy(dct["stationarities"])
            for k, v in stationarities_dict.items():
                stationarities_dict[k] = Stationarity[v.upper()]

            return DatasetMetadata(
                dataset_id=tuple(dct["dataset_id"]),  # type: ignore
                is_train=dct["is_train"],
                length=dct["length"],
                dimensions=dct["dimensions"],
                contamination=dct["contamination"],
                anomaly_length=anomaly_length,
                num_anomalies=dct["num_anomalies"],
                means=dct["means"],
                stddevs=dct["stddevs"],
                trends=trends,
                stationarities=stationarities_dict,
            )
        else:
            return dct
