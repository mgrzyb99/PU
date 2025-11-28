import os
import sys
import warnings
from argparse import ArgumentParser

import pandas as pd
from plot_results import preprocess_data

sys.path.append(".")

from modules.names import COST_FUNCTION_NAMES, DATASET_NAMES, TARGET_NAMES
from modules.utils import query_aim_repo

ROOT = "results/"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("metric")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--subset", default="val")
    parser.add_argument("--mode", default="max")
    args = parser.parse_args()

    if args.repo is None:
        args.repo = os.path.join(ROOT, args.experiment)

    warnings.filterwarnings(
        action="ignore", message=".*DataFrame is highly fragmented.*"
    )
    warnings.filterwarnings(
        action="ignore",
        message=".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
    )

    data = query_aim_repo(
        path=args.repo,
        experiment=args.experiment,
        metric=args.metric,
        subset=args.subset,
    )
    data = preprocess_data(data, experiment=args.experiment)

    if args.mode == "max":
        data = data.loc[data.groupby("run.hash")["value"].idxmax()]
    elif args.mode == "min":
        data = data.loc[data.groupby("run.hash")["value"].idxmin()]
    else:
        raise ValueError("Wrong mode value")

    match args.experiment:
        case "cost_functions":
            groupby_columns = ["dataset", "cost_function"]
            unstack_level = "dataset"
            row_order = [
                value
                for value in COST_FUNCTION_NAMES.values()
                if value in set(data["cost_function"])
            ]
            col_order = [
                value
                for value in DATASET_NAMES.values()
                if value in set(data["dataset"])
            ]
        case "imbalanced":
            groupby_columns = ["target", "cost_family", "learning_method"]
            unstack_level = "target"
            row_order = [
                ("Sigmoid", "nnPU ($\\hat{g}(x) > 0$)"),
                ("Sigmoid", "Imba. nnPU"),
                ("Logistic", "nnPU ($\\hat{g}(x) > 0$)"),
                ("Logistic", "nnPU ($\\hat{\\eta}(x) > \\pi$)"),
                ("Logistic", "Log. adj. nnPU"),
                ("Logistic", "Imba. nnPU"),
            ]
            col_order = [
                value
                for value in TARGET_NAMES["CIFAR10"].values()
                if value in set(data["target"])
            ]
        case "mixup":
            groupby_columns = ["dataset", "mixup_loss", "mixup_gamma"]
            unstack_level = "dataset"
            row_order = [
                ("None", 0.0),
                ("Chen", 0.1),
                ("Chen", 0.3),
                ("Chen", 1.0),
                ("Chen", 3.0),
                ("Zhao", 0.1),
                ("Zhao", 0.3),
                ("Zhao", 1.0),
                ("Zhao", 3.0),
            ]
            col_order = [
                value
                for value in DATASET_NAMES.values()
                if value in set(data["dataset"])
            ]
        case _:
            raise NotImplementedError("Experiment not implemented")

    data = (
        data.groupby(groupby_columns)
        .aggregate({"value": ["mean", "std"]})
        .dropna(axis="index", how="all")
    )

    data = (
        (
            "\\("
            + data[("value", "mean")].map("{:.3f}".format).str.lstrip("0")
            + "\\pm"
            + data[("value", "std")].map("{:.3f}".format).str.lstrip("0")
            + "\\)"
        )
        .unstack(unstack_level)
        .reindex(row_order)[col_order]
    )

    if isinstance(data.index, pd.MultiIndex):
        data.index.names = [
            name.replace("_", " ").capitalize() for name in data.index.names
        ]
    else:
        data.index.name = data.index.name.replace("_", " ").capitalize()

    if args.experiment == "cost_functions":
        data.index.set_names("Loss function", inplace=True)

    if args.experiment == "imbalanced":
        data.index.set_names({"Cost family": "Loss function"}, inplace=True)

    if args.experiment == "mixup":
        data.rename({0.0: "-"}, level="Mixup gamma", inplace=True)
        data.index.set_names(
            {"Mixup loss": "Mixup method", "Mixup gamma": "\\(\\gamma\\)"}, inplace=True
        )

    data.to_csv(sys.stdout)
