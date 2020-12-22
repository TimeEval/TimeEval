from timeeval import TimeEval
from timeeval.timeeval import Algorithm
from timeeval.adapters.jar import JarAdapter
from timeeval.adapters.distributed import DistributedAdapter
from timeeval.utils.window import reverse_windowing
from timeeval.utils.convert2bin import numpy2bin
import json
import numpy as np
from pathlib import Path


args = json.load(open("dads.json", "r"))
remotes = json.load(open("remotes.json", "r")).get("remotes", list())


def prepare_data(_: str, data: np.ndarray) -> np.ndarray:
    numpy2bin(data, Path(args.get("sequence")))
    return data


class DADS(JarAdapter):
    def _postprocess_data(self, data: np.ndarray) -> np.ndarray:
        w1 = reverse_windowing(data, args.get("query-length") + 1)
        w2 = reverse_windowing(w1, args.get("sub-sequence-length"))
        w3 = reverse_windowing(w2, args.get("sub-sequence-length") + 5)
        return w3


dads = DADS("/home/phillip.wenig/Projects/DADS/target/distributed-anomaly-detection.jar",
            "results.txt",
            ["master"], args)

dads = DistributedAdapter(dads, "java -jar /home/phillip.wenig/Projects/DADS/target/distributed-anomaly-detection.jar slave --host $HOSTNAME --master-host odin01 --no-statistics", "phillip.wenig", remote_hosts=[r.get("host") for r in remotes])

timeeval = TimeEval([f"taxi.{i}" for i in range(1, 10)], [Algorithm(name="dads", data_as_file=False, function=dads)],
                    prepare_data=prepare_data,
                    dataset_config=Path("datasets.json"))
timeeval.run()

print(timeeval.results)
json.dump(timeeval.results, open("results.json", "w"))
