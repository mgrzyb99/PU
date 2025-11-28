from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, KMNIST, MNIST, FashionMNIST, VisionDataset
from torchvision.transforms import v2

from .dataset import CachedDatasetWrapper, PNDatasetWrapper, PUDatasetWrapper


class DataModuleBase(LightningDataModule, ABC):
    def __init__(
        self,
        *,
        train_dataloader_kwargs: dict[str, Any] = {},
        val_dataloader_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.train_dataloader_kwargs = {
            "shuffle": True,
            "drop_last": True,
        } | train_dataloader_kwargs
        self.val_dataloader_kwargs = {
            "shuffle": False,
            "drop_last": False,
        } | val_dataloader_kwargs

    @abstractmethod
    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def setup(self, stage: str) -> None:
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)


class TVDatasetMixin:
    dataset_cls: type[VisionDataset]
    transform: Callable | None

    def __init__(self, *args, root: str = ".data/", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.root = root

    def prepare_data(self) -> None:
        super().prepare_data()
        self.dataset_cls(root=self.root, train=True, download=True)
        self.dataset_cls(root=self.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == "fit":
            self.train_dataset = self.dataset_cls(
                root=self.root, train=True, transform=self.transform
            )
            self.val_dataset = self.dataset_cls(
                root=self.root, train=False, transform=self.transform
            )
        else:
            raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        return self.dataset_cls.__name__


class PNDatasetMixin:
    def __init__(
        self,
        *args,
        pn_wrap_kwargs: dict[str, Any] = {},
        pn_resample_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pn_wrap_kwargs = pn_wrap_kwargs
        self.pn_resample_kwargs = pn_resample_kwargs

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == "fit":
            self.train_dataset = PNDatasetWrapper.wrap(
                self.train_dataset, **self.pn_wrap_kwargs
            )
            if self.pn_resample_kwargs is not None:
                if (
                    "positive_count" in self.pn_resample_kwargs
                    and "negative_count" in self.pn_resample_kwargs
                ):
                    self.train_dataset = self.train_dataset.resample_using_counts(
                        **self.pn_resample_kwargs
                    )
                elif (
                    "positive_prior" in self.pn_resample_kwargs
                    or "dataset_length" in self.pn_resample_kwargs
                ):
                    self.train_dataset = self.train_dataset.resample_using_params(
                        **self.pn_resample_kwargs
                    )
                else:
                    raise ValueError("wrong `pn_resample_kwargs` value")
            self.val_dataset = PNDatasetWrapper.wrap(
                self.val_dataset, **self.pn_wrap_kwargs
            )
        else:
            raise NotImplementedError


class PUDatasetMixin:
    def __init__(self, *args, pu_wrap_kwargs: dict[str, Any] = {}, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pu_wrap_kwargs = pu_wrap_kwargs

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == "fit":
            if (
                "positive_count" in self.pu_wrap_kwargs
                and "unlabeled_count" in self.pu_wrap_kwargs
            ):
                self.train_dataset = PUDatasetWrapper.wrap_using_counts(
                    self.train_dataset, **self.pu_wrap_kwargs
                )
            elif "label_frequency" in self.pu_wrap_kwargs:
                self.train_dataset = PUDatasetWrapper.wrap_using_params(
                    self.train_dataset, **self.pu_wrap_kwargs
                )
            else:
                raise ValueError("wrong `pu_wrap_kwargs` value")
        else:
            raise NotImplementedError


class CachedDatasetMixin:
    def __init__(
        self, *args, cache_data: bool = True, prefetch_data: bool = True, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache_data = cache_data
        self.prefetch_data = prefetch_data

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == "fit":
            if self.cache_data:
                self.train_dataset = CachedDatasetWrapper(self.train_dataset)
                self.val_dataset = CachedDatasetWrapper(self.val_dataset)
                if self.prefetch_data:
                    for index in range(len(self.train_dataset)):
                        self.train_dataset[index]
                    for index in range(len(self.val_dataset)):
                        self.val_dataset[index]
        else:
            raise NotImplementedError


class PNMNISTDataModule(
    PNDatasetMixin, CachedDatasetMixin, TVDatasetMixin, DataModuleBase
):
    dataset_cls = MNIST
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )


class PUMNISTDataModule(PUDatasetMixin, PNMNISTDataModule):
    pass


class PNFMNISTDataModule(
    PNDatasetMixin, CachedDatasetMixin, TVDatasetMixin, DataModuleBase
):
    dataset_cls = FashionMNIST
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.2860], std=[0.3530]),
        ]
    )


class PUFMNISTDataModule(PUDatasetMixin, PNFMNISTDataModule):
    pass


class PNKMNISTDataModule(
    PNDatasetMixin, CachedDatasetMixin, TVDatasetMixin, DataModuleBase
):
    dataset_cls = KMNIST
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1918], std=[0.3483]),
        ]
    )


class PUKMNISTDataModule(PUDatasetMixin, PNKMNISTDataModule):
    pass


class PNCIFAR10DataModule(
    PNDatasetMixin, CachedDatasetMixin, TVDatasetMixin, DataModuleBase
):
    dataset_cls = CIFAR10
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )


class PUCIFAR10DataModule(PUDatasetMixin, PNCIFAR10DataModule):
    pass
