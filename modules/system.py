from abc import ABC, abstractmethod

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.distributions import Beta
from torch.utils.data import Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    MulticlassAccuracy,
)

from .label import PN_LABELS, Label
from .loss import (
    CostFunctionBase,
    ImbalanceduPURiskEstimator,
    PNRiskEstimator,
    RiskEstimatorBase,
    uPURiskEstimator,
)
from .mixup import ChenMixupLoss, MixupLossBase


class SystemBase(LightningModule, ABC):
    _positive_threshold = None
    risk_estimator: RiskEstimatorBase

    def __init__(
        self,
        model: nn.Module,
        *,
        positive_threshold: float | None = 0.5,
        mixup_loss: MixupLossBase | None = None,
        mixup_alpha: float = 1.0,
        mixup_gamma: float = 1.0,
        log_on_step: bool = False,
    ) -> None:
        assert mixup_alpha > 0.0
        assert mixup_gamma > 0.0
        super().__init__()
        self.model = model
        if positive_threshold is not None:
            self.positive_threshold = positive_threshold
        self.mixup_loss = mixup_loss
        self.mixup_alpha = mixup_alpha
        self.mixup_gamma = mixup_gamma
        self.log_on_step = log_on_step
        self.mixup_dist = Beta(mixup_alpha, mixup_alpha)
        self.val_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "f1_score": BinaryF1Score(),
                "balanced_accuracy": MulticlassAccuracy(num_classes=2, average="macro"),
            },
            prefix="val_",
        )
        self.val_auroc = BinaryAUROC()

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.model(samples))

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return self.cost_function.predict(scores)

    @abstractmethod
    def erm_loss(
        self, positive_risk: torch.Tensor, negative_risk: torch.Tensor
    ) -> torch.Tensor:
        pass

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        samples, targets = batch
        scores = self.forward(samples)
        positive_risk, negative_risk = self.risk_estimator(scores, targets)
        erm_loss = self.erm_loss(positive_risk, negative_risk)
        log_dict = {
            "train_positive_risk": positive_risk,
            "train_negative_risk": negative_risk,
            "train_erm_loss": erm_loss,
        }
        if self.mixup_loss is not None:
            samples_1, probas_1 = samples, self.predict(scores)
            if isinstance(self.mixup_loss, ChenMixupLoss):
                probas_1 = torch.where(targets == Label.POSITIVE, 1.0, probas_1)
            samples_2, probas_2 = (
                torch.roll(samples_1, 1, 0),
                torch.roll(probas_1, 1, 0),
            )
            lam = self.mixup_dist.sample()
            samples_m = lam * samples_1 + (1.0 - lam) * samples_2
            scores_m = self.forward(samples_m)
            probas_m = self.predict(scores_m)
            targets_m = lam * probas_1 + (1.0 - lam) * probas_2
            mixup_loss = self.mixup_loss(probas_m, targets_m)
            log_dict["train_mixup_loss"] = mixup_loss
            loss = erm_loss + self.mixup_gamma * mixup_loss
        else:
            loss = erm_loss
        self.log_dict(log_dict, on_step=self.log_on_step, on_epoch=not self.log_on_step)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        samples, targets = batch
        scores = self.forward(samples)
        probas = self.predict(scores)
        labels = torch.where(probas > self.positive_threshold, *PN_LABELS)
        self.val_metrics.update(labels, targets)
        self.val_auroc.update(probas, targets)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.log("val_auroc", self.val_auroc.compute())
        self.val_metrics.reset()
        self.val_auroc.reset()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            hparams = {
                "system": {
                    "dataset_name": self.dataset_name,
                    "train_dataset": {
                        "positive_prior": self.train_dataset.positive_prior
                    },
                    "val_dataset": {"positive_prior": self.val_dataset.positive_prior},
                }
            }
            if hasattr(self.train_dataset, "label_frequency"):
                hparams["system"]["train_dataset"]["label_frequency"] = (
                    self.train_dataset.label_frequency
                )
            if hasattr(self.risk_estimator, "positive_prior"):
                if self.risk_estimator.positive_prior is None:
                    self.risk_estimator.positive_prior = (
                        self.train_dataset.positive_prior
                    )
                hparams["system"]["risk_estimator"] = {
                    "positive_prior": self.risk_estimator.positive_prior
                }
            if hasattr(self.cost_function, "positive_prior"):
                if self.cost_function.positive_prior is None:
                    self.cost_function.positive_prior = (
                        self.train_dataset.positive_prior
                    )
                hparams["system"]["cost_function"] = {
                    "positive_prior": self.cost_function.positive_prior
                }
            if self.positive_threshold is None:
                self.positive_threshold = self.train_dataset.positive_prior
            hparams["system"]["positive_threshold"] = self.positive_threshold
            self.logger.log_hyperparams(hparams)
        else:
            raise NotImplementedError

    @property
    def positive_threshold(self) -> float | None:
        return self._positive_threshold

    @positive_threshold.setter
    def positive_threshold(self, value: float) -> None:
        assert 0.0 < value < 1.0
        self._positive_threshold = value

    @property
    def cost_function(self) -> CostFunctionBase:
        return self.risk_estimator.cost_function

    @property
    def dataset_name(self) -> str:
        return self.trainer.datamodule.dataset_name

    @property
    def train_dataset(self) -> Dataset:
        return self.trainer.datamodule.train_dataset

    @property
    def val_dataset(self) -> Dataset:
        return self.trainer.datamodule.val_dataset


class PNSystem(SystemBase):
    def __init__(
        self,
        *args,
        cost_function: CostFunctionBase,
        positive_prior: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.risk_estimator = PNRiskEstimator(
            cost_function, positive_prior=positive_prior
        )

    def erm_loss(
        self, positive_risk: torch.Tensor, negative_risk: torch.Tensor
    ) -> torch.Tensor:
        erm_loss = positive_risk + negative_risk
        return erm_loss


class uPUSystem(SystemBase):  # noqa: N801
    def __init__(
        self,
        *args,
        cost_function: CostFunctionBase,
        positive_prior: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.risk_estimator = uPURiskEstimator(
            cost_function, positive_prior=positive_prior
        )

    def erm_loss(
        self, positive_risk: torch.Tensor, negative_risk: torch.Tensor
    ) -> torch.Tensor:
        erm_loss = positive_risk + negative_risk
        return erm_loss


class nnPUSystem(SystemBase):  # noqa: N801
    def __init__(
        self,
        *args,
        cost_function: CostFunctionBase,
        positive_prior: float | None = None,
        nnpu_beta: float = 0.0,
        nnpu_gamma: float = 1.0,
        **kwargs,
    ) -> None:
        assert nnpu_beta >= 0.0
        assert 0.0 < nnpu_gamma <= 1.0
        super().__init__(*args, **kwargs)
        self.risk_estimator = uPURiskEstimator(
            cost_function, positive_prior=positive_prior
        )
        self.nnpu_beta = nnpu_beta
        self.nnpu_gamma = nnpu_gamma

    def erm_loss(
        self, positive_risk: torch.Tensor, negative_risk: torch.Tensor
    ) -> torch.Tensor:
        if negative_risk >= -self.nnpu_beta:
            erm_loss = positive_risk + negative_risk
            step_direction = 1.0
        else:
            erm_loss = -self.nnpu_gamma * negative_risk
            step_direction = -1.0
        self.log(
            "train_step_direction",
            step_direction,
            on_step=self.log_on_step,
            on_epoch=not self.log_on_step,
        )
        return erm_loss


class ImbalancednnPUSystem(SystemBase):
    def __init__(
        self,
        *args,
        cost_function: CostFunctionBase,
        positive_prior: float | None = None,
        expected_prior: float = 0.5,
        nnpu_beta: float = 0.0,
        nnpu_gamma: float = 1.0,
        **kwargs,
    ) -> None:
        assert nnpu_beta >= 0.0
        assert 0.0 < nnpu_gamma <= 1.0
        super().__init__(*args, **kwargs)
        self.risk_estimator = ImbalanceduPURiskEstimator(
            cost_function, positive_prior=positive_prior, expected_prior=expected_prior
        )
        self.nnpu_beta = nnpu_beta
        self.nnpu_gamma = nnpu_gamma

    def erm_loss(
        self, positive_risk: torch.Tensor, negative_risk: torch.Tensor
    ) -> torch.Tensor:
        if negative_risk >= -self.nnpu_beta:
            erm_loss = positive_risk + negative_risk
            step_direction = 1.0
        else:
            erm_loss = -self.nnpu_gamma * negative_risk
            step_direction = -1.0
        self.log(
            "train_step_direction",
            step_direction,
            on_step=self.log_on_step,
            on_epoch=not self.log_on_step,
        )
        return erm_loss
