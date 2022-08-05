# How to integrate your own algorithm into TimeEval

```{warning}
WIP
```

If your algorithm is written in Python, you could use our {class}`~timeeval.adapters.function.FunctionAdapter` ([Example](../concepts/algorithms.md#function-adapter) of using the `FunctionAdapter`).
However, this comes with some limitations (such as no way to limit resource usage or setting timeouts).
We, therefore, highly recommend to use the {class}`~timeeval.adapters.docker.DockerAdapter`.
This means that we have to create a Docker image for your algorithm before we can use it in TimeEval.

In the following, we assume that we want to create a Docker image with your algorithm to execute it with TimeEval.
We provide base images for various programming languages.
You can find them [here](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).

```{note}
Please contact the maintainers if there is no base image for your algorithm programming language or runtime.
```

## Procedure

1. Build base image
   1. Clone the timeeval-algorithms repository
   2. Build the selected base image from [`0-base-images`](https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/0-base-images).
      Please make sure that you tag your image correctly (the image name must match the `FROM`-clause in your algorithm image.

2. Integrate your algorithm into TimeEval and build the Docker image (you can use any algorithm in this repository as an example for that)
   - TimeEval uses a common interface to execute all its algorithms (using the `DockerAdapter`).
     This interface describes data input and output as well as algorithm configuration.
     The calling-interface is described in this repositories' [README](https://github.com/HPI-Information-Systems/TimeEval-algorithms#timeeval-algorithm-interface).
     Please read the section carefully and adapt your algorithm to the interface description.
     You could also create a wrapper script that takes care of the integration.
     Our canonical file format for time series datasets is described [here](https://github.com/HPI-Information-Systems/TimeEval#canonical-file-format).
   - Create a `Dockerfile` for your algorithm that is based on your selected base image ([example](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/kmeans/Dockerfile)).
   - Build your algorithm Docker image.
   - Check if your algorithm can read a time series using our common file format.
   - Check if the algorithm parameters are correctly set using TimeEval's call format.
   - Check if the anomaly scores are written in the correct format (an anomaly score value for each point of the original time series in a headerless CSV-file).
   - The README contains [example calls](https://github.com/HPI-Information-Systems/TimeEval-algorithms#example-calls) to test your algorithm after you have build the Docker image for it.

3. Install TimeEval (`pip install timeeval==1.2.4`)

4. Create an experiment script with your configuration of datasets, algorithms, etc.
