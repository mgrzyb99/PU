from abc import ABC, abstractmethod
from functools import cache
from typing import Any, Self

import numpy
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset

from .label import PN_LABELS, PU_LABELS, Label, check_labels


class DatasetWrapperBase(Dataset, ABC):
    def __init__(
        self, dataset: Dataset, *, indices: ArrayLike, targets: ArrayLike
    ) -> None:
        assert len(indices) == len(targets)
        self.dataset = dataset
        self.indices = numpy.array(indices)
        self.targets = numpy.array(targets)

    @classmethod
    @abstractmethod
    def wrap(cls, dataset: Dataset) -> Self:
        pass

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        old_index = self.indices[index]
        sample, _ = self.dataset[old_index]
        target = self.targets[index]
        return sample, target

    def __len__(self) -> int:
        return len(self.indices)


def get_targets(dataset: Dataset) -> NDArray:
    if hasattr(dataset, "targets"):
        return numpy.array(dataset.targets)
    else:
        return numpy.array([target for _, target in dataset])


class PNDatasetWrapper(DatasetWrapperBase):
    def __init__(
        self,
        dataset: Dataset,
        *,
        indices: ArrayLike,
        targets: ArrayLike,
        positive_labels: ArrayLike,
        included_labels: ArrayLike,
    ) -> None:
        assert check_labels(targets, PN_LABELS)
        super().__init__(dataset, indices=indices, targets=targets)
        self.positive_labels = numpy.array(positive_labels)
        self.included_labels = numpy.array(included_labels)

    @classmethod
    def wrap(
        cls,
        dataset: Dataset,
        *,
        positive_labels: ArrayLike,
        included_labels: ArrayLike | None = None,
    ) -> Self:
        indices = numpy.arange(len(dataset))
        targets = get_targets(dataset)
        if included_labels is not None:
            is_included = numpy.isin(targets, included_labels)
            indices = indices[is_included]
            targets = targets[is_included]
        else:
            included_labels = numpy.unique(targets)
        is_positive = numpy.isin(targets, positive_labels)
        targets = numpy.where(is_positive, *PN_LABELS)
        return cls(
            dataset,
            indices=indices,
            targets=targets,
            positive_labels=positive_labels,
            included_labels=included_labels,
        )

    def resample_using_counts(
        self,
        *,
        positive_count: int,
        negative_count: int,
        random_seed: int | None = None,
    ) -> Self:
        assert positive_count > 0
        assert negative_count > 0
        is_positive = self.targets == Label.POSITIVE
        if random_seed is not None:
            numpy.random.seed(random_seed)
        positive_indices = numpy.random.choice(
            self.indices[is_positive], positive_count, replace=False
        )
        negative_indices = numpy.random.choice(
            self.indices[~is_positive], negative_count, replace=False
        )
        indices = numpy.concatenate((positive_indices, negative_indices))
        targets = numpy.repeat(PN_LABELS, (positive_count, negative_count))
        return self.__class__(
            self.dataset,
            indices=indices,
            targets=targets,
            positive_labels=self.positive_labels,
            included_labels=self.included_labels,
        )

    def resample_using_params(
        self,
        *,
        positive_prior: float | None = None,
        dataset_length: int | float = 1.0,
        random_seed: int | None = None,
    ) -> Self:
        if positive_prior is None:
            positive_prior = self.positive_prior
        if isinstance(dataset_length, float):
            dataset_length = dataset_length * len(self)
        positive_count = round(positive_prior * dataset_length)
        negative_count = round((1.0 - positive_prior) * dataset_length)
        return self.resample_using_counts(
            positive_count=positive_count,
            negative_count=negative_count,
            random_seed=random_seed,
        )

    @property
    def positive_prior(self) -> float:
        return (numpy.sum(self.targets == Label.POSITIVE) / len(self)).item()


def get_positive_prior(
    dataset: Dataset, *, positive_label: Any = Label.POSITIVE
) -> float:
    if hasattr(dataset, "positive_prior"):
        return dataset.positive_prior
    else:
        return (numpy.sum(get_targets(dataset) == positive_label) / len(dataset)).item()


class PUDatasetWrapper(DatasetWrapperBase):
    def __init__(
        self,
        dataset: Dataset,
        *,
        indices: ArrayLike,
        targets: ArrayLike,
        positive_label: Any,
    ) -> None:
        assert check_labels(targets, PU_LABELS)
        super().__init__(dataset, indices=indices, targets=targets)
        self.positive_label = positive_label

    @classmethod
    def wrap(cls, dataset: Dataset) -> Self:
        raise NotImplementedError(
            "use `wrap_using_counts` or `wrap_using_params` instead"
        )

    @classmethod
    def wrap_using_counts(
        cls,
        dataset: Dataset,
        *,
        positive_count: int,
        unlabeled_count: int,
        positive_label: Any = Label.POSITIVE,
        random_seed: int | None = None,
    ) -> Self:
        assert positive_count > 0
        assert unlabeled_count > 0
        indices = numpy.arange(len(dataset))
        targets = get_targets(dataset)
        is_positive = targets == positive_label
        if random_seed is not None:
            numpy.random.seed(random_seed)
        positive_indices = numpy.random.choice(
            indices[is_positive], positive_count, replace=False
        )
        unlabeled_indices = numpy.random.choice(indices, unlabeled_count, replace=False)
        indices = numpy.concatenate((positive_indices, unlabeled_indices))
        targets = numpy.repeat(PU_LABELS, (positive_count, unlabeled_count))
        return cls(
            dataset, indices=indices, targets=targets, positive_label=positive_label
        )

    @classmethod
    def wrap_using_params(
        cls,
        dataset: Dataset,
        *,
        label_frequency: float,
        dataset_length: int | float = 1.0,
        positive_label: Any = Label.POSITIVE,
        random_seed: int | None = None,
    ) -> Self:
        if isinstance(dataset_length, float):
            dataset_length = dataset_length * len(dataset)
        positive_prior = get_positive_prior(dataset, positive_label=positive_label)
        scaling_factor = 1.0 / (1.0 - label_frequency * (1.0 - positive_prior))
        positive_count = round(
            scaling_factor * label_frequency * positive_prior * dataset_length
        )
        unlabeled_count = round(
            scaling_factor * (1.0 - label_frequency) * dataset_length
        )
        return cls.wrap_using_counts(
            dataset,
            positive_count=positive_count,
            unlabeled_count=unlabeled_count,
            positive_label=positive_label,
            random_seed=random_seed,
        )

    @property
    def positive_prior(self) -> float:
        return get_positive_prior(self.dataset, positive_label=self.positive_label)

    @property
    def label_frequency(self) -> float:
        return (
            numpy.sum(self.targets == Label.POSITIVE)
            / numpy.sum(get_targets(self.dataset)[self.indices] == self.positive_label)
        ).item()


class CachedDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @cache
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def targets(self) -> Any:
        return self.dataset.targets
