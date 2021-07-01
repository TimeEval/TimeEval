from typing import Final, List
from pathlib import Path


METRICS_CSV = "metrics.csv"
EXECUTION_LOG = "execution.log"
ANOMALY_SCORES_TS = "anomaly_scores.ts"
HYPER_PARAMETERS = "hyper_params.json"
RESULTS_CSV = "results.csv"


class HPI_CLUSTER:
    odin01: Final[str] = "odin01"
    odin02: Final[str] = "odin02"
    odin03: Final[str] = "odin03"
    odin04: Final[str] = "odin04"
    odin05: Final[str] = "odin05"
    odin06: Final[str] = "odin06"
    odin07: Final[str] = "odin07"
    odin08: Final[str] = "odin08"
    thor01: Final[str] = "thor01"
    thor02: Final[str] = "thor02"
    thor03: Final[str] = "thor03"
    thor04: Final[str] = "thor04"

    odin01_ip: Final[str] = "172.16.64.61"
    odin02_ip: Final[str] = "172.16.64.62"
    odin03_ip: Final[str] = "172.16.64.63"
    odin04_ip: Final[str] = "172.16.64.64"
    odin05_ip: Final[str] = "172.16.64.65"
    odin06_ip: Final[str] = "172.16.64.66"
    odin07_ip: Final[str] = "172.16.64.67"
    odin08_ip: Final[str] = "172.16.64.68"
    thor01_ip: Final[str] = "172.16.64.55"
    thor02_ip: Final[str] = "172.16.64.56"
    thor03_ip: Final[str] = "172.16.64.57"
    thor04_ip: Final[str] = "172.16.64.58"

    nodes: Final[List[str]] = [
        odin01, odin02, odin03, odin04, odin05, odin06, odin07, odin08, thor01, thor02, thor03, thor04
    ]
    nodes_ip: Final[List[str]] = [
        odin01_ip, odin02_ip, odin03_ip, odin04_ip, odin05_ip, odin06_ip, odin07_ip, odin08_ip,
        thor01_ip, thor02_ip, thor03_ip, thor04_ip
    ]

    akita_benchmark_path: Final[Path] = Path("/home/projects/akita/data/benchmark-data/data-processed")
    akita_test_case_path: Final[Path] = Path("/home/projects/akita/data/test-cases")
