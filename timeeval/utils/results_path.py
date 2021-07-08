from pathlib import Path


def generate_experiment_path(base_results_dir: Path, algorithm_name: str, hyper_params_id: str, collection_name: str, dataset_name: str, repetition_number: int) -> Path:
    return base_results_dir / algorithm_name / hyper_params_id / collection_name / dataset_name / str(repetition_number)
