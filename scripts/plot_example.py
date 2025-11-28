import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects

SEED = 42

MEAN_POSITIVE = (-1, -1)
COV_POSITIVE = ((1, 0.5), (0.5, 1))
N_POSITIVE = 100

MEAN_NEGATIVE = (1, 1)
COV_NEGATIVE = ((1, -0.5), (-0.5, 1))
N_NEGATIVE = 100

N_LABELED = 50
assert N_LABELED < N_POSITIVE

CONTEXT = "paper"
STYLE = "whitegrid"
FONT_SCALE = 1.5

FIGSIZE = (6.4, 6.4)
MARKER_SIZE = 200
ASPECT = 1.5

COLOR_POSITIVE = "tab:green"
MARKER_POSITIVE = "o"

COLOR_NEGATIVE = "tab:red"
MARKER_NEGATIVE = "X"

COLOR_UNLABELED = "tab:gray"

FONTSIZE = 20
STROKE_WIDTH = 5
STROKE_COLOR = "white"

np.random.seed(SEED)

x = np.concat(
    (
        np.random.multivariate_normal(MEAN_POSITIVE, COV_POSITIVE, N_POSITIVE),
        np.random.multivariate_normal(MEAN_NEGATIVE, COV_NEGATIVE, N_NEGATIVE),
    ),
    0,
)
y = np.repeat((1, -1), (N_POSITIVE, N_NEGATIVE))

s = np.full(N_POSITIVE + N_NEGATIVE, -1)
s[np.random.choice(N_POSITIVE, N_LABELED)] = 1

df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y, "s": s}).sort_values(
    ["y", "s"]
)

sns.set_theme(context=CONTEXT, style=STYLE, font_scale=FONT_SCALE)

for plot_type in ("pn", "pu"):
    fig = plt.figure(figsize=FIGSIZE)
    ax = sns.scatterplot(
        df,
        x="x1",
        y="x2",
        hue="y" if plot_type == "pn" else "s",
        style="y",
        palette={
            1: COLOR_POSITIVE,
            -1: COLOR_NEGATIVE if plot_type == "pn" else COLOR_UNLABELED,
        },
        markers={1: MARKER_POSITIVE, -1: MARKER_NEGATIVE},
        legend=False,
        s=MARKER_SIZE,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)

    for label_type in ("p", "n"):
        txt = plt.text(
            *(MEAN_POSITIVE if label_type == "p" else MEAN_NEGATIVE),
            (
                "Positive"
                if label_type == "p"
                else ("Negative" if plot_type == "pn" else "Unlabeled")
            ),
            color=(
                COLOR_POSITIVE
                if label_type == "p"
                else (COLOR_NEGATIVE if plot_type == "pn" else COLOR_UNLABELED)
            ),
            fontsize=FONTSIZE,
            weight="bold",
            ha="center",
            va="center",
        )
        txt.set_path_effects(
            [patheffects.withStroke(linewidth=STROKE_WIDTH, foreground=STROKE_COLOR)]
        )

    fig.tight_layout()
    fig.savefig(f"figures/{plot_type}_data.pdf")
    fig.savefig(f"figures/{plot_type}_data.png")
