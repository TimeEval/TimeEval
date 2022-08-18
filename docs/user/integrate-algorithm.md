# How to integrate your own algorithm into TimeEval

If your algorithm is written in Python, you could use our {class}`~timeeval.adapters.function.FunctionAdapter` ([Example](../concepts/algorithms.md#function-adapter) of using the `FunctionAdapter`).
However, this comes with some limitations (such as no way to limit resource usage or setting timeouts).
We, therefore, highly recommend to use the {class}`~timeeval.adapters.docker.DockerAdapter`.
This means that you have to create a Docker image for your algorithm before you can use it in TimeEval.

In the following, we assume that you want to create a Docker image with your algorithm to execute it with TimeEval.
We provide base images for various programming languages.
You can find them [here](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).

## Procedure

This section contains a short guide on how to integrate your own algorithm into TimeEval **using the `DockerAdapter`**.
There are three main steps:
(i) Preparing the base image, (ii) creating the algorithm image, and (iii) using the algorithm image within TimeEval.

### (i) Prepare the base image

1. Clone the [`timeeval-algorithms`-repository](https://github.com/HPI-Information-Systems/TimeEval-algorithms)
2. Build the selected base image from [`0-base-images`](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).
   Please make sure that you tag your image correctly (the image name must match the `FROM`-clause in your algorithm image).

   - change to the `0-base-images` folder: `cd 0-base-images`
   - build your desired base image, e.g. `docker build -t registry.gitlab.hpi.de/akita/i/python3-base:0.2.5 ./python3-base`
   - (optionally: build derived base image, e.g. `docker build -t registry.gitlab.hpi.de/akita/i/pyod:0.2.5 ./pyod`)
   - now you can build your algorithm image from the base image (see next section)

```{note}
Please contact the maintainers if there is no base image for your algorithm programming language or runtime.
```

### (ii) Integrate your algorithm into TimeEval by creating an algorithm image

You can use any algorithm in the `timeeval-algorithms`-repository as an example for that.

TimeEval uses a common interface to execute all its Docker algorithms.
This interface describes data input and output as well as algorithm configuration.
The calling-interface is described in [](../concepts/algorithms.md#timeeval-algorithm-interface).
Please read the section carefully and adapt your algorithm to the interface description.
You could also create a wrapper script that takes care of the integration.
Our canonical file format for time series datasets is described [here](../concepts/datasets.md#canonical-file-format).
Once you are familiar with the concepts, you can adapt your algorithm and create its Docker image:

1. Create a `Dockerfile` for your algorithm that is based on your selected base image.
   Example:

   ```Dockerfile
   FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

   LABEL maintainer="sebastian.schmidl@hpi.de"

   ENV ALGORITHM_MAIN="/app/algorithm.py"

   # install algorithm dependencies
   COPY requirements.txt /app/
   RUN pip install -r /app/requirements.txt

   # add algorithm implementation
   COPY algorithm.py /app/
   ```

2. Build your algorithm Docker image, e.g. `docker build -t my_algorithm:latest Dockerfile`

3. Check if your algorithm is compatible to TimeEval.

   - Check if your algorithm can read a time series using our common file format.
   - Check if the algorithm parameters are correctly set using TimeEval's call format.
   - Check if the anomaly scores are written in the correct format (an anomaly score value for each point of the original time series in a headerless CSV-file).

   The [README](https://github.com/HPI-Information-Systems/TimeEval-algorithms#usage) of the [`timeeval-algorithms`-repository](https://github.com/HPI-Information-Systems/TimeEval-algorithms) provides further details and instructions on how to create and test TimeEval algorithm images, including example calls.

### (iii) Use algorithm image within TimeEval

Create an experiment script with your configuration of datasets and your own algorithm image.
Make sure that you specify your algorithm's image and tag name correctly and use `skip_pull=True`.
This prevents TimeEval from trying to update your algorithm image by fetching it from a Docker registry because your image was not published to any registry.
In addition, `data_as_file` must also be enabled for all algorithms using the `DockerAdapter`.
Please also specify the algorithm's learning type (whether it requires training and which training data) and input dimensionality (uni- or multivariate):

```python
#!/usr/bin/env python3

from pathlib import Path

from timeeval import TimeEval, MultiDatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import FixedParameters


def main():
    dm = MultiDatasetManager([Path("<datasets-folder>")])
    dataset = dm.select()

    ####################
    # Add your own algorithm
    ####################
    algorithms = [Algorithm(
        name="<my_algorithm>",
        main=DockerAdapter(
            image_name="<my_algorithm>",
            tag="latest",
            skip_pull=True  # must be set to True because your image is just available locally
        ),
        # Set your custom parameters:
        param_config=FixedParameters({
            "random_state": 42,
            # ...
        }),
        # required by DockerAdapter
        data_as_file=True,
        # set the following metadata based on your algorithm:
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.MULTIVARIATE
    )]


    timeeval = TimeEval(dm, datasets, algorithms)
    timeeval.run()
    results = timeeval.get_results()
    print(results)


if __name__ == "__main__":
    main()
```
