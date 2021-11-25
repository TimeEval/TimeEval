from typing import List, Union
import numpy as np
import pandas as pd

from pathlib import Path
from jinja2 import Environment, PackageLoader

from .config import result_file_path, algo_cite_lut


class LatexPlotGenerator():

    def __init__(self, result_file: Path, quality_metric: str = "ROC_AUC", dominant_aggregation: str = "mean") -> None:
        self._jenv = Environment(
            loader=PackageLoader("latex_generator", "templates"),
            keep_trailing_newline=True,
            lstrip_blocks=True,
            trim_blocks=True,
            block_start_string="((*", block_end_string="*))",
            variable_start_string="((", variable_end_string="))",
            comment_start_string="((#", comment_end_string="#))",
        )
        # load results
        print(f"Reading results from {result_file.resolve()}")
        self.df = pd.read_csv(result_file)
        self.df_asl = None
        self.df_asl = self._extract_asl(self.df, quality_metric, dominant_aggregation)

    def _extract_asl(self, df: pd.DataFrame, quality_metric: str = "ROC_AUC", dominant_aggregation: str = "mean") -> pd.DataFrame:
        index_columns = ["algo_input_dimensionality", "algo_training_type", "algorithm"]

        if quality_metric == "ROC_AUC" and dominant_aggregation == "mean" and self.df_asl is not None and df.equals(self.df):
            return self.df_asl

        df_asl = df.pivot(index=index_columns, columns=["collection", "dataset"], values=quality_metric)
        df_asl = df_asl.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df_asl[dominant_aggregation] = df_asl.agg(dominant_aggregation, axis=1)
        df_asl = df_asl.reset_index().sort_values(by=index_columns[:-1] + [dominant_aggregation], ascending=False).set_index(index_columns)
        df_asl = df_asl.drop(columns=dominant_aggregation)
        return df_asl

    def _extract_overview_table(self, relative_rates: bool = True) -> pd.DataFrame:
        df_algorithm_error_counts = self.df.pivot_table(index=["algorithm"], columns=["error_category"], values="repetition", aggfunc="count")
        df_algorithm_error_counts = df_algorithm_error_counts.fillna(value=0).astype(np.int64)
        error_categories = [c for c in df_algorithm_error_counts.columns if not c.startswith("-")]
        df_algorithm_error_counts["- ERROR -"] = df_algorithm_error_counts[error_categories].sum(axis=1)
        df_algorithm_error_counts = df_algorithm_error_counts.drop(columns=error_categories)
        df_algorithm_error_counts["- ALL -"] = df_algorithm_error_counts.sum(axis=1)
        df_algorithm_error_counts.columns = [c.split(" ")[1] for c in df_algorithm_error_counts.columns]

        def get_error_count(algo, tpe="ERROR"):
            if relative_rates:
                return df_algorithm_error_counts.loc[algo, tpe] / df_algorithm_error_counts.loc[algo, "ALL"]
            else:
                return df_algorithm_error_counts.loc[algo, tpe]
            
        overview = []
        for d, l, a in self.df_asl.index:
            overview.append([d,l,a])
        df_overview_table = pd.DataFrame(overview, columns=["dimensionality", "learning type", "algorithm"])
        df_overview_table["# TIMEOUT"] = df_overview_table["algorithm"].apply(get_error_count, tpe="TIMEOUT")
        df_overview_table["# OOM"] = df_overview_table["algorithm"].apply(get_error_count, tpe="OOM")
        df_overview_table["# ERROR"] = df_overview_table["algorithm"].apply(get_error_count, tpe="ERROR")
        return df_overview_table
    
    @staticmethod
    def _compute_boxplot_values(algo_name: str, values: np.ndarray) -> dict:
        values = np.sort(values)
        q1 = np.quantile(values, 0.25)
        q3 = np.quantile(values, 0.75)
        iqr = q3 - q1
        return {
            "name": algo_name,
            "lower_whisker": values[values >= q1-1.5*iqr][0],
            "upper_whisker": values[values <= q3+1.5*iqr][-1],
            "lower_quartile": q1,
            "upper_quartile": q3,
            "median": np.median(values),
            "average": np.mean(values),
            "sample_size": len(values)
        }
    
    @staticmethod
    def _adapt_levels_and_add_rules(algorithms: List[dict], attr: str = "dim", rulespan: str = "1-6", limit_factor = 2):
        def add_multirow_and_rule(algo: dict, count: int, add_rule: bool = False):
            # get original text
            text = algo[attr]
            # shorten by limitting factor
            text = text[:(limit_factor*count)-1]
            # add multirow and rotate
            text = f"\multirow{{{count}}}{{*}}{{\myrotate{{{text}}}}}"
            algo[attr] = text
            # potentially add a midrule
            if add_rule and not algo["dim"].startswith("\cmidrule"):
                algo["dim"] = f"\cmidrule[0.25pt](l){{{rulespan}}}\n    " + algo["dim"]

        first_algo = None
        current_value = None
        count = 0
        for a in algorithms:
            if a[attr] != current_value:
                if first_algo is not None:
                    add_multirow_and_rule(first_algo, count, add_rule=first_algo != algorithms[0])

                first_algo = a
                count = 1
                current_value = a[attr]
            else:
                a[attr] = ""
                count += 1
        if first_algo is not None:
            add_multirow_and_rule(first_algo, count, add_rule=True)
    
    def generate_table(self, target: Union[str, Path], relative_rates: bool = True) -> None:
        # Prepare target
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Load template and data
        file_template = self._jenv.get_template("table.tex.jinja2")
        df_overview_table = self._extract_overview_table(relative_rates)

        # format and prepare algorithms (entries in the table)
        def format(f: float) -> str:
            if relative_rates:
                return "{:02.0f}~\%".format(f*100)
            else:
                return "{:03.0f}".format(f)

        algorithms = []
        for _, s in df_overview_table.iterrows():
            algorithms.append({
                "name": algo_cite_lut[s["algorithm"]],
                "timeout": format(s["# TIMEOUT"]),
                "oom": format(s["# OOM"]),
                "error": format(s["# ERROR"]),
                "dim": s["dimensionality"],
                "learn": s["learning type"].replace("_", "-"),
            })
        
        self._adapt_levels_and_add_rules(algorithms, "dim", rulespan=("1-6"))
        self._adapt_levels_and_add_rules(algorithms, "learn", rulespan=("2-6"))

        # Write latex using template
        with target_path.open("w") as fh:
            fh.write(file_template.render(
                algorithms=algorithms
            ))

    def generate_boxplot(self, target: Union[str, Path], show_outliers: bool = False) -> None:
        # Prepare target
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Load template
        file_template = self._jenv.get_template("tikz-boxplot.tex.jinja2")

        boxplots= []
        for index, s in self.df_asl.iterrows():
            algo_name = index[-1]
            values = s.dropna().values
            boxplots.append(self._compute_boxplot_values(algo_name, values))

        # Write latex using template
        with target_path.open("w") as fh:
            fh.write(file_template.render(
                title="ROC-AUC all datasets",
                show_outliers=show_outliers,
                box_extend=0.8,
                algo_boxplots=boxplots
            ))
    
    def generate_overview_table(self, target: Union[str, Path], relative_rates: bool = True, metrics: List[str] = ["ROC_AUC"], include_gutentag_only: bool = False) -> None:
        # Prepare target
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Load template and data
        file_template = self._jenv.get_template("overview-table.tex.jinja2")
        df_overview_table = self._extract_overview_table(relative_rates)
        asls = {}
        titles = [f"{m.replace('_', '-')} all datasets" for m in metrics]
        for metric in metrics:
            asls[metric] = self._extract_asl(self.df, quality_metric=metric, dominant_aggregation="mean")
        if include_gutentag_only:
            asls["gutentag"] = self._extract_asl(self.df[self.df["collection"] == "GutenTAG"], quality_metric="ROC_AUC", dominant_aggregation="mean")
            titles.append("ROC-AUC GutenTAG only")

        # format and prepare algorithms (entries in the table)
        def format(f: float) -> str:
            if relative_rates:
                return "{:02.0f}~\%".format(f*100)
            else:
                return "{:03.0f}".format(f)

        algorithms = []
        for _, s in df_overview_table.iterrows():
            boxplots = []
            gt_boxplot = None
            for metric in metrics:
                values = asls[metric].loc[(s["dimensionality"], s["learning type"], s["algorithm"])].dropna().values
                boxplots.append(self._compute_boxplot_values(s["algorithm"], values))
            if include_gutentag_only:
                try:
                    values = asls["gutentag"].loc[(s["dimensionality"], s["learning type"], s["algorithm"])].dropna().values
                    gt_boxplot = self._compute_boxplot_values(s["algorithm"], values)
                except Exception:
                    pass
            algorithms.append({
                "name": algo_cite_lut[s["algorithm"]],
                "timeout": format(s["# TIMEOUT"]),
                "oom": format(s["# OOM"]),
                "error": format(s["# ERROR"]),
                "dim": s["dimensionality"],
                "learn": s["learning type"].replace("_", "-"),
                "boxes": boxplots,
                "gt_box": gt_boxplot
            })
        
        self._adapt_levels_and_add_rules(algorithms, "dim", rulespan=("1-6"))
        self._adapt_levels_and_add_rules(algorithms, "learn", rulespan=("2-6"))

        # Write latex using template
        with target_path.open("w") as fh:
            fh.write(file_template.render(
                algorithms=algorithms,
                boxplot_titles=titles,
                gt_boxplots=include_gutentag_only,
                box_extend=0.8,
            ))


def main():
    gen = LatexPlotGenerator(result_file_path)
    gen.generate_overview_table("overview-table.tex", metrics=[
        "ROC_AUC",
        "PR_AUC",
        # "RANGE_PR_AUC"
        ], include_gutentag_only=True)
    

if __name__ == "__main__":
    main()
