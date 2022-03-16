from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_pcc_parameters: Dict[str, Dict[str, Any]] = {
 "max_iter": {
  "defaultValue": None,
  "description": "Number of iterations for the power method computed by `svd_solver == 'randomized'`.",
  "name": "max_iter",
  "type": "int"
 },
 "n_components": {
  "defaultValue": None,
  "description": "Number of components to keep. If `n_components` is not set all components are kept: `n_components == min(n_samples, n_features)`.",
  "name": "n_components",
  "type": "int"
 },
 "n_selected_components": {
  "defaultValue": None,
  "description": "Number of selected principal components for calculating the outlier scores. It is not necessarily equal to the total number of the principal components. If not set, use all principal components.",
  "name": "n_selected_components",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Used when `svd_solver == 'arpack' or svd_solver == 'randomized'` to seed random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "svd_solver": {
  "defaultValue": "auto",
  "description": "'auto': the solver is selected by a default policy based on `X.shape` and `n_components`. If the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient 'randomized' method is enabled. Otherwise the exact full SVD is computed and optionally truncated afterwards. 'full': run exact full SVD calling the standard LAPACK solver via `scipy.linalg.svd` and select the components by postprocessing. 'arpack': run SVD truncated to n_components calling ARPACK solver via `scipy.sparse.linalg.svds`. It requires strictly `0 < n_components < X.shape[1]`. 'randomized': run randomized SVD by the method of Halko et al.",
  "name": "svd_solver",
  "type": "enum[auto,full,arpack,randomized]"
 },
 "tol": {
  "defaultValue": 0.0,
  "description": "Tolerance for singular values computed by `svd_solver == 'arpack'`.",
  "name": "tol",
  "type": "float"
 },
 "whiten": {
  "defaultValue": "False",
  "description": "When True the `components_` vectors are multiplied by the square root of `n_samples` and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.",
  "name": "whiten",
  "type": "boolean"
 }
}


def pcc(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="PCC",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/pcc",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_pcc_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
