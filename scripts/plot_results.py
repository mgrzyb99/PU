import os
import sys
import warnings
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns

sys.path.append(".")

from modules.names import (
    COST_FUNCTION_NAMES,
    DATASET_NAMES,
    METRIC_NAMES,
    MIXUP_LOSS_NAMES,
    RISK_ESTIMATOR_NAMES,
    SUBSET_NAMES,
    TARGET_NAMES,
)
from modules.utils import query_aim_repo

ROOT = "results/"

CONTEXT = "paper"
STYLE = "whitegrid"
PALETTE = "bright"
FONT_SCALE = 1.5

COL_WRAP = 2
HEIGHT = 4.8
ASPECT = 4 / 3
ESTIMATOR = "mean"
ERRORBAR = "sd"
LINEWIDTH = 2


def preprocess_data(data: pd.DataFrame, experiment: str) -> pd.DataFrame:
    data["step"] += 1
    data["epoch"] += 1
    data["dataset"] = data["run.hparams.system.dataset_name"].map(DATASET_NAMES)
    data["risk_estimator"] = (
        data["run.hparams.config.model.class_path"]
        .str.split(".")
        .str.get(-1)
        .map(RISK_ESTIMATOR_NAMES)
    )
    data["cost_function"] = (
        data["run.hparams.config.model.init_args.cost_function.class_path"]
        .str.split(".")
        .str.get(-1)
        .map(COST_FUNCTION_NAMES)
    )
    if experiment == "imbalanced":
        data["threshold"] = (
            data["run.hparams.config.model.init_args.positive_threshold"]
            .fillna("\\pi")
            .astype(str)
        )
        data.loc[
            (data["risk_estimator"] == "nnPU")
            & ~(data["cost_function"].str.contains("Log. adj."))
            & (data["threshold"] == "0.5"),
            "learning_method",
        ] = "nnPU ($\\hat{g}(x) > 0$)"
        data.loc[
            (data["risk_estimator"] == "nnPU")
            & ~(data["cost_function"].str.contains("Log. adj."))
            & (data["threshold"] == "\\pi"),
            "learning_method",
        ] = "nnPU ($\\hat{\\eta}(x) > \\pi$)"
        data.loc[
            (data["risk_estimator"] == "nnPU")
            & (data["cost_function"].str.contains("Log. adj.")),
            "learning_method",
        ] = "Log. adj. " + data["risk_estimator"]
        data.loc[data["risk_estimator"] == "Imba. nnPU", "learning_method"] = data[
            "risk_estimator"
        ]
        data["cost_family"] = (
            data["cost_function"]
            .str.removeprefix("Log. adj.")
            .str.lstrip()
            .str.capitalize()
        )
        data["target"] = data[
            "run.hparams.config.data.init_args.pn_wrap_kwargs.positive_labels"
        ].map(TARGET_NAMES["CIFAR10"])
    if experiment == "mixup":
        data["mixup_loss"] = (
            data["run.hparams.config.model.init_args.mixup_loss.class_path"]
            .str.split(".")
            .str.get(-1)
            .map(MIXUP_LOSS_NAMES, na_action="ignore")
            .fillna("None")
        )
        data["mixup_gamma"] = data["run.hparams.config.model.init_args.mixup_gamma"]
        data.loc[data["mixup_loss"] == "None", "mixup_gamma"] = 0.0
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("metric")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--subset", default="val")
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

    relplot_args = {
        "x": "epoch",
        "y": "value",
        "col_wrap": COL_WRAP,
        "kind": "line",
        "height": HEIGHT,
        "aspect": ASPECT,
        "estimator": ESTIMATOR,
        "errorbar": ERRORBAR,
        "linewidth": LINEWIDTH,
        "facet_kws": {"legend_out": False},
    }

    match args.experiment:
        case "kiryo+":
            relplot_args |= {
                "hue": "risk_estimator",
                "col": "dataset",
                "hue_order": [
                    value
                    for value in RISK_ESTIMATOR_NAMES.values()
                    if value in set(data["risk_estimator"])
                ],
                "col_order": [
                    value
                    for value in DATASET_NAMES.values()
                    if value in set(data["dataset"])
                ],
            }
        case "cost_functions":
            relplot_args |= {
                "hue": "cost_function",
                "col": "dataset",
                "hue_order": [
                    value
                    for value in COST_FUNCTION_NAMES.values()
                    if value in set(data["cost_function"])
                ],
                "col_order": [
                    value
                    for value in DATASET_NAMES.values()
                    if value in set(data["dataset"])
                ],
            }
        case "imbalanced":
            relplot_args |= {
                "hue": "learning_method",
                "style": "cost_family",
                "col": "target",
                "hue_order": [
                    "nnPU ($\\hat{g}(x) > 0$)",
                    "nnPU ($\\hat{\\eta}(x) > \\pi$)",
                    "Log. adj. nnPU",
                    "Imba. nnPU",
                ],
                "style_order": ["Sigmoid", "Logistic"],
                "col_order": [
                    value
                    for value in TARGET_NAMES["CIFAR10"].values()
                    if value in set(data["target"])
                ],
            }
        case "mixup":
            relplot_args |= {
                "hue": "mixup_gamma",
                "style": "mixup_loss",
                "col": "dataset",
                "style_order": ["None", "Chen", "Zhao"],
                "col_order": [
                    value
                    for value in DATASET_NAMES.values()
                    if value in set(data["dataset"])
                ],
                "palette": [(0.0, 0.0, 0.0), *sns.color_palette(PALETTE, n_colors=4)],
            }
        case _:
            raise NotImplementedError("Experiment not implemented")

    sns.set_theme(context=CONTEXT, style=STYLE, palette=PALETTE, font_scale=FONT_SCALE)

    grid = sns.relplot(data, **relplot_args)
    grid.despine(left=True, bottom=True)

    grid.set_titles(
        f"{relplot_args['col'].replace('_', ' ').capitalize()} = {{col_name}}"
    )
    grid.set_xlabels(relplot_args["x"].replace("_", " ").capitalize())
    grid.set_ylabels(
        f"{METRIC_NAMES[args.metric]} ({SUBSET_NAMES[args.subset].lower()})"
    )

    sns.move_legend(
        grid,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=len(grid.legend.get_lines()),
        title=grid.legend.get_title().get_text().replace("_", " ").capitalize(),
    )

    if args.experiment == "cost_functions":
        grid.legend.set_title("Loss function")

    if args.experiment == "imbalanced":
        handles, labels = grid.axes[0].get_legend_handles_labels()
        grid.legend.remove()

        grid.figure.legend(
            handles[1:5],
            labels[1:5],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=4,
            frameon=False,
            title="Learning method",
        )

        grid.figure.legend(
            handles[6:8],
            labels[6:8],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.1),
            ncols=2,
            frameon=False,
            title="Loss function",
        )

    if args.experiment == "mixup":
        handles, labels = grid.axes[0].get_legend_handles_labels()
        grid.legend.remove()

        grid.figure.legend(
            handles[2:6],
            labels[2:6],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=4,
            frameon=False,
            title="$\\gamma$ (regularization strength)",
        )

        grid.figure.legend(
            handles[7:10],
            labels[7:10],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.1),
            ncols=3,
            frameon=False,
            title="Mixup method",
        )

    grid.savefig(f"figures/{args.experiment}_{args.metric}.pdf")
    grid.savefig(f"figures/{args.experiment}_{args.metric}.png")
