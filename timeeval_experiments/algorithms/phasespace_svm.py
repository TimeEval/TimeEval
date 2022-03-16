from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_phasespace_svm_parameters: Dict[str, Dict[str, Any]] = {
 "coef0": {
  "defaultValue": 0.0,
  "description": "Independent term in kernel function. It is only significant in \u2018poly\u2019 and \u2018sigmoid\u2019.",
  "name": "coef0",
  "type": "float"
 },
 "degree": {
  "defaultValue": 3,
  "description": "Degree of the polynomial kernel function (\u2018poly\u2019). Ignored by all other kernels.",
  "name": "degree",
  "type": "int"
 },
 "embed_dim_range": {
  "defaultValue": [
   50,
   100,
   150
  ],
  "description": "List of phase space dimensions (sliding window sizes). For each dimension a OC-SVM is fitted to calculate outlier scores. The final result is the point-wise aggregation of the anomaly scores.",
  "name": "embed_dim_range",
  "type": "List[int]"
 },
 "gamma": {
  "defaultValue": None,
  "description": "Kernel coefficient for \u2018rbf\u2019, \u2018poly\u2019 and \u2018sigmoid\u2019. If gamma is not set (`None`) then it uses 1 / (n_features * X.var()) as value of gamma",
  "name": "gamma",
  "type": "float"
 },
 "kernel": {
  "defaultValue": "rbf",
  "description": "Specifies the kernel type to be used in the algorithm. It must be one of \u2018linear\u2019, \u2018poly\u2019, \u2018rbf\u2019, or \u2018sigmoid\u2019.",
  "name": "kernel",
  "type": "enum[linear,poly,rbf,sigmoid]"
 },
 "nu": {
  "defaultValue": 0.5,
  "description": "Main parameter of OC-SVM. An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1].",
  "name": "nu",
  "type": "float"
 },
 "project_phasespace": {
  "defaultValue": "False",
  "description": "Whether to use phasespace projection or just work on the phasespace values.",
  "name": "project_phasespace",
  "type": "boolean"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "tol": {
  "defaultValue": 0.001,
  "description": "Tolerance for stopping criterion.",
  "name": "tol",
  "type": "float"
 }
}


def phasespace_svm(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="PhaseSpace-SVM",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/phasespace_svm",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_phasespace_svm_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
