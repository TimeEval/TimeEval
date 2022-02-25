from typing import List
from pathlib import Path


METRICS_CSV = "metrics.csv"
EXECUTION_LOG = "execution.log"
ANOMALY_SCORES_TS = "anomaly_scores.ts"
HYPER_PARAMETERS = "hyper_params.json"
RESULTS_CSV = "results.csv"


class HPI_CLUSTER:
    odin01: str = "odin01"
    odin02: str = "odin02"
    odin03: str = "odin03"
    odin04: str = "odin04"
    odin05: str = "odin05"
    odin06: str = "odin06"
    odin07: str = "odin07"
    odin08: str = "odin08"
    thor01: str = "thor01"
    thor02: str = "thor02"
    thor03: str = "thor03"
    thor04: str = "thor04"
    thor05: str = "thor05"
    thor06: str = "thor06"

    odin01_ip: str = "172.16.64.61"
    odin02_ip: str = "172.16.64.62"
    odin03_ip: str = "172.16.64.63"
    odin04_ip: str = "172.16.64.64"
    odin05_ip: str = "172.16.64.65"
    odin06_ip: str = "172.16.64.66"
    odin07_ip: str = "172.16.64.67"
    odin08_ip: str = "172.16.64.68"
    thor01_ip: str = "172.16.64.55"
    thor02_ip: str = "172.16.64.56"
    thor03_ip: str = "172.16.64.57"
    thor04_ip: str = "172.16.64.58"
    thor05_ip: str = "172.16.64.82"
    thor06_ip: str = "172.16.64.83"

    odin_nodes: List[str] = [
        odin01, odin02, odin03, odin04, odin05, odin06, odin07, odin08
    ]
    thor_nodes: List[str] = [thor01, thor02, thor03, thor04]
    thor_ext_nodes: List[str] = [thor05, thor06]
    nodes: List[str] = odin_nodes + thor_nodes + thor_ext_nodes

    odin_nodes_ip: List[str] = [
        odin01_ip, odin02_ip, odin03_ip, odin04_ip, odin05_ip, odin06_ip, odin07_ip, odin08_ip
    ]
    thor_nodes_ip: List[str] = [thor01_ip, thor02_ip, thor03_ip, thor04_ip]
    thor_ext_nodes_ip: List[str] = [thor05_ip, thor06_ip]
    nodes_ip: List[str] = odin_nodes_ip + thor_nodes_ip + thor_ext_nodes_ip

    akita_benchmark_path: Path = Path("/home/projects/akita/data/benchmark-data/data-processed")
    akita_test_case_path: Path = Path("/home/projects/akita/data/test-cases")
    akita_correlation_anomalies_path: Path = Path("/home/projects/akita/data/correlation-anomalies")
    akita_test_case_variable_length_path: Path = Path("/home/projects/akita/data/variable-length")
