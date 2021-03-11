import argparse
import logging
import shutil
from functools import reduce
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


class ResultMerger:
    KEY_COLUMNS = ["algorithm", "collection", "dataset", "repetition", "hyper_params_id"]

    def __init__(self, left_folder: Path, right_folder: Path, target_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.target_folder = target_folder

    def prepare_target_folder(self) -> None:
        if self.left_folder == self.target_folder or self.right_folder == self.target_folder:
            raise ValueError("The merge result can only be stored in a new empty folder for safety reasons!")
        if self.target_folder.exists():
            if not list(self.target_folder.iterdir()):
                self.target_folder.rmdir()
            else:
                raise ValueError("Target folder exists and is not empty!")
        shutil.copytree(self.left_folder, self.target_folder)

    def _copy_run(self, run: pd.Series, overwrite: bool = False) -> None:
        subfolder = Path(".") / run["algorithm"] / run["hyper_params_id"] / run["collection"] / run["dataset"] / str(run["repetition"])
        if overwrite:
            shutil.rmtree(self.target_folder / subfolder)
        shutil.copytree(self.right_folder / subfolder, self.target_folder / subfolder)

    @staticmethod
    def _build_selection_mask(ref: pd.DataFrame, record: pd.Series) -> pd.Series:
        masks = []
        for c in ResultMerger.KEY_COLUMNS:
            masks.append(ref[c] == record[c])
        return reduce(lambda x, y: np.logical_and(x, y), masks)  # type: ignore

    def copy_misc_files(self) -> None:
        misc_files = [f for f in self.right_folder.iterdir() if f.is_file() and f.name != "results.csv"]
        for file in misc_files:
            target_filename = self.target_folder / file.name
            # rename file if the name already exists in the target folder:
            counter = 0
            while target_filename.exists():
                counter += 1
                target_filename = self.target_folder / f"{file.stem}-{counter}{file.suffix}"

            logging.info(f"Copying additional file {target_filename.name}")
            shutil.copy2(file, target_filename)

    def _append(self, df_ref: pd.DataFrame, record: pd.Series, change_filesystem: bool = False) -> pd.DataFrame:
        new_idx = len(df_ref)
        logging.info(
            f"Run ({record[self.KEY_COLUMNS].values}) not in reference experiment; adding it as {new_idx}.")
        if change_filesystem:
            self._copy_run(record, overwrite=False)
        return df_ref.append(record, ignore_index=True)

    def _replace(self, df_ref: pd.DataFrame, idx: int, record: pd.Series, change_filesystem: bool = False) -> None:
        logging.info(f"Found replacement for reference experiment {idx} ({record.values})"
                     f"with new status {record['status']}; replacing it")
        if change_filesystem:
            self._copy_run(record, overwrite=True)
        df_ref.iloc[idx] = record

    def merge_runs(self, change_filesystem: bool = False) -> Tuple[pd.DataFrame, List[int]]:
        df_ref = pd.read_csv(self.left_folder / "results.csv")
        df_right = pd.read_csv(self.right_folder / "results.csv")
        changed_idxs = []
        for i, record in df_right.iterrows():
            mask = ResultMerger._build_selection_mask(df_ref, record)
            ref_indices = df_ref[mask].index
            if len(ref_indices) == 0:
                df_ref = self._append(df_ref, record, change_filesystem)
                changed_idxs.append(len(df_ref) - 1)
            elif len(ref_indices) == 1:
                idx = ref_indices[0]
                ref_status = df_ref.loc[idx, "status"]
                right_status = record["status"]
                if ref_status == "Status.ERROR" and (right_status == "Status.OK" or right_status == "Status.TIMEOUT"):
                    self._replace(df_ref, idx, record, change_filesystem)
                    changed_idxs.append(idx)
                elif ref_status == "Status.TIMEOUT" and right_status == "Status.OK":
                    self._replace(df_ref, idx, record, change_filesystem)
                    changed_idxs.append(idx)
                # skip all others
            else:
                logging.warning(f"There are ambiguous matched for reference indices {ref_indices} and right index {i}!"
                                f"This should not happen; skipping those!")

        if change_filesystem:
            df_ref.to_csv(self.target_folder / "results.csv")
        return df_ref, changed_idxs

    def merge(self):
        self.prepare_target_folder()
        _, changed_idxs = self.merge_runs(change_filesystem=True)
        self.copy_misc_files()
        logging.info(f"Added {len(changed_idxs)} runs from {self.right_folder} to {self.left_folder}."
                     f"Result stored at {self.target_folder}")


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge results of two different experiments together.")
    parser.add_argument("left", type=Path,
                        help="Folder of reference experiment. Only erroneous or timed-out runs are replaced with runs "
                             "from `right`; new runs in `right` are appended to this.")
    parser.add_argument("right", type=Path,
                        help="Folder of another experiment containing additional runs that will be merged to `left`.")
    parser.add_argument("target", type=Path,
                        help="Path where the merged results are written to.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _create_arg_parser()
    ResultMerger(args.left, args.right, args.target).merge()
