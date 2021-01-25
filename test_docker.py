from timeeval import TimeEval, Algorithm, Datasets
from timeeval.adapters import DockerAdapter
from pathlib import Path


algorithm = Algorithm(name="test-docker", main=DockerAdapter("algorithm-template", Path("results/"), {}), data_as_file=True)

timeeval = TimeEval(Datasets("tests/example_data/"), [("test", "dataset-datetime")], [algorithm])
timeeval.run()
print(timeeval.results)

# todo: delete file
