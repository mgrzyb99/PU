from collections.abc import Mapping, Sequence
from typing import Any

import aim
import pandas as pd


def flatten(
    mapping: Mapping[str, Any], old_key: str = "", key_sep: str = "."
) -> dict[str, Any]:
    items = []
    for key, value in mapping.items():
        new_key = old_key + key_sep + key if old_key else key
        if isinstance(value, Mapping):
            items.extend(flatten(value, new_key, key_sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def query_aim_repo(
    path: str, experiment: str, metric: str, subset: str
) -> pd.DataFrame:
    repo = aim.Repo(path)
    try:
        data = pd.concat(
            (
                metric_to_dataframe(metric)
                for metric in repo.query_metrics(
                    f"run.experiment == '{experiment}' and \
                    metric.name == '{metric}' and \
                    metric.context.subset == '{subset}'"
                )
            ),
            ignore_index=True,
        )
    except ValueError as e:
        raise RuntimeError("Empty query result") from e
    finally:
        repo.close()
    return data


def metric_to_dataframe(metric: aim.Metric) -> pd.DataFrame:
    data = metric.dataframe()
    data["metric.name"] = metric.name
    for key, value in flatten(metric.context.to_dict(), "metric.context").items():
        data[key] = value
    data["run.hash"] = metric.run.hash
    data["run.experiment"] = metric.run.experiment
    for key, value in flatten(metric.run["hparams"], "run.hparams").items():
        if isinstance(value, Sequence):
            value = str(value)
        data[key] = value
    return data
