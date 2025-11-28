from enum import IntEnum

import numpy
import torch
from numpy.typing import ArrayLike


class Label(IntEnum):
    POSITIVE = 1
    NEGATIVE = 0
    UNLABELED = -1


PN_LABELS = (Label.POSITIVE, Label.NEGATIVE)
PU_LABELS = (Label.POSITIVE, Label.UNLABELED)


def check_labels(labels: ArrayLike | torch.Tensor, test_labels: ArrayLike) -> bool:
    if isinstance(labels, torch.Tensor):
        return torch.all(
            torch.isin(labels, torch.tensor(test_labels, device=labels.device))
        ).item()
    else:
        return numpy.all(numpy.isin(labels, test_labels)).item()
