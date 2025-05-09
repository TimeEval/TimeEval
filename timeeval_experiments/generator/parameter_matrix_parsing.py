import json
import re
from distutils.util import strtobool
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Union, Any, Dict, List

import pandas as pd


LIST_TYPETAG_PATTERN = re.compile(r"list\[.*\]", re.I)


class ParameterCategory(Enum):
    FIXED = "fixed"
    DEPENDENT = "dependent"
    SHARED = "shared"
    OPTIMIZED = "optimized"


class ParameterImpact(Enum):
    QUALITY = "quality"
    PERFORMANCE = "performance"


class ParameterMatrixProxy:

    def __init__(self, parameter_matrix_path: Union[Path, str]):
        df = self._load_parameter_matrix(parameter_matrix_path)
        self._params_df = self._get_params_df(df)
        self._algorithms = self._extract_algorithms(df)

    @staticmethod
    def _load_parameter_matrix(path: Union[Path, str]) -> pd.DataFrame:
        path = Path(path)
        df = pd.read_csv(path, header=None)
        df = df.drop(columns=[1])
        return df

    @staticmethod
    def _get_params_df(df: pd.DataFrame) -> pd.DataFrame:
        df_params = df.iloc[0:6].T
        headers = df_params.iloc[0]
        df_params = df_params.iloc[1:]
        df_params.columns = list(map(lambda x: x.lower(), headers))
        return df_params

    @staticmethod
    def _extract_algorithms(df: pd.DataFrame) -> Dict[str, List[str]]:
        df_only_algos = df.iloc[6:]
        algo_names = df_only_algos[0]
        df_algos = df_only_algos.apply(lambda s: s[~pd.isnull(s)].map(lambda x: algo_names[s.name]), axis=1)
        df_algos.columns = df.iloc[0]
        df_algos = df_algos.drop(columns=["name"])

        params_algo_mapping = {}
        for param in df_algos.columns:
            s = df_algos[param]
            if len(s.shape) == 2:
                for _ in range(s.shape[1] - 1):
                    s.iloc[:, 0].fillna(s.iloc[:, -1], inplace=True)
                    s = s.iloc[:, :-1]
                s = s[param]
            params_algo_mapping[param] = list(s[~pd.isnull(s)].unique())

        return params_algo_mapping

    @staticmethod
    def _parse_type(value: str, type_tag: str) -> Any:
        if value == "default" or value == "MANUAL":
            return value
        else:
            if type_tag.lower() == "int":
                return int(float(value.replace(",", "")))
            elif type_tag.lower() == "float":
                return float(value.replace(",", ""))
            elif type_tag.lower() == "boolean":
                return bool(strtobool(value))
            elif LIST_TYPETAG_PATTERN.match(type_tag):
                return json.loads(value)
            else:
                return value

    def fixed_params(self) -> Dict[str, Any]:
        fixed_df: pd.DataFrame = self._params_df.loc[self._params_df["category"] == ParameterCategory.FIXED.value, ["name", "type", "value range"]]
        fp = {}
        for _, (name, tpe, value) in fixed_df.iterrows():
            fp[name] = self._parse_type(str(value), tpe)
        return fp

    def shared_params(self) -> Dict[str, Dict[str, Any]]:
        df_shared: pd.DataFrame = self._params_df.loc[self._params_df["category"] == ParameterCategory.SHARED.value, ["name", "type", "value range"]]
        sp = {}
        for _, (name, tpe, value) in df_shared.iterrows():
            sp[name] = {
                "algorithms": self._algorithms[name],
                # "type": tpe,
                "search_space": json.loads(value)
            }
        return sp

    def dependent_params(self) -> Dict[str, Union[str, List[str]]]:
        dependent_df: pd.DataFrame = self._params_df.loc[self._params_df["category"] == ParameterCategory.DEPENDENT.value, ["name", "value range"]]
        dp = {}
        for _, (name, value) in dependent_df.iterrows():
            if value.startswith('[') and value.endswith(']'):  # e.g. ["heuristic 1", "heuristic 2"]
                dp[name] = json.loads(value)
            else:  # e.g. heuristic 1
                dp[name] = value
        return dp

    def optimized_params(self) -> Dict[str, Any]:
        optimizted_df: pd.DataFrame = self._params_df.loc[self._params_df["category"] == ParameterCategory.OPTIMIZED.value, ["name", "value range", "count"]]

        op: Dict[str, Union[str, Dict[str, List[Any]]]] = {}
        for _, (name, value, count) in optimizted_df.iterrows():
            try:
                value = json.loads(value)
            except JSONDecodeError:
                pass
            if int(count) > 1:
                dd = {}
                for algo in self._algorithms[name]:
                    dd[algo] = value
                op[name] = dd
            else:
                op[name] = value

        return op


if __name__ == "__main__":
    pmp = ParameterMatrixProxy("timeeval_experiments/parameter-matrix.csv")
    print("Fixed parameters:")
    print(pmp.fixed_params())
    print("")
    print("Shared parameters:")
    print(pmp.shared_params())
    print("")
    print("Dependent parameters:")
    print(pmp.dependent_params())
    print("")
    print("Optimized parameters:")
    print(pmp.optimized_params())
