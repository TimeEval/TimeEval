# Algorithm Code Generator

This module can generate the algorithm call functions for the algorithms from the [TimeEval Algorithms repo](https://gitlab.hpi.de/akita/timeeval-algorithms) and the parameter settings from the parameter matrix file.

## Usage

1. Install the `TimeEval` project
   1. `python setup.py install`
2. Run the generator command as module
   1. `python -m timeeval_experiments.generator [algo-stubs|param-config] ...`
3. For help or parameter details call
   1. `python -m timeeval_experiments.generator -h`

### Example

Generate the parameter configurations
```
python -m timeeval_experiments.generator param-config --output param-config.json parameter-matrix.csv
```

Generate the algorithms code
```
python -m timeeval_experiments.generator algo-stubs ~/Projects/timeeval-algorithms
```
