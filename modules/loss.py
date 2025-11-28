from abc import ABC, abstractmethod

import torch
from torch import nn

from .label import PN_LABELS, PU_LABELS, Label, check_labels


class CostFunctionBase(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def polarize_labels(labels: torch.Tensor) -> torch.Tensor:
        assert check_labels(labels, PN_LABELS)
        return torch.where(labels == Label.POSITIVE, 1.0, -1.0)


class SigmoidCostFunction(CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return 1.0 / (1.0 + torch.exp(scores * targets))

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.where(scores > 0.0, 1.0, 0.0)


class ExponentialCostFunction(CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return torch.exp(-scores * targets)

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-2.0 * scores))


class LogisticCostFunction(CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return torch.log(1.0 + torch.exp(-scores * targets))

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-scores))


class SavageCostFunction(CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return 1.0 / (1.0 + torch.exp(scores * targets)) ** 2

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-scores))


class TangentCostFunction(CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return (2.0 * torch.atan(scores * targets) - 1.0) ** 2

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.atan(scores) + 0.5, 0.0, 1.0)


class PositivePriorMixin:
    _positive_prior = None

    def __init__(self, *args, positive_prior: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if positive_prior is not None:
            self.positive_prior = positive_prior

    @property
    def positive_prior(self) -> float | None:
        return self._positive_prior

    @positive_prior.setter
    def positive_prior(self, value: float) -> None:
        assert 0.0 < value < 1.0
        self._positive_prior = value


class LogitAdjustedLogisticCostFunction(PositivePriorMixin, CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return torch.log(
            1.0
            + (((1.0 - self.positive_prior) / self.positive_prior) ** targets)
            * torch.exp(-scores * targets)
        )

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-scores))


class LogitAdjustedSavageCostFunction(PositivePriorMixin, CostFunctionBase):
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self.polarize_labels(targets)
        return (
            1.0
            / (
                1.0
                + ((self.positive_prior / (1.0 - self.positive_prior)) ** targets)
                * torch.exp(scores * targets)
            )
            ** 2
        )

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-scores))


class RiskEstimatorBase(nn.Module, ABC):
    def __init__(self, cost_function: CostFunctionBase) -> None:
        super().__init__()
        self.cost_function = cost_function

    @abstractmethod
    def forward(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def empirical_risk(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        assert scores.shape == targets.shape
        if scores.numel() > 0:
            return torch.mean(self.cost_function(scores, targets))
        else:
            return torch.tensor(0.0)


class PNRiskEstimator(PositivePriorMixin, RiskEstimatorBase):
    def forward(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert check_labels(targets, PN_LABELS)
        is_positive = targets == Label.POSITIVE
        positive_risk = self.positive_prior * self.empirical_risk(
            scores[is_positive], targets[is_positive]
        )
        negative_risk = (1.0 - self.positive_prior) * self.empirical_risk(
            scores[~is_positive], targets[~is_positive]
        )
        return positive_risk, negative_risk


class uPURiskEstimator(PositivePriorMixin, RiskEstimatorBase):  # noqa: N801
    def forward(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert check_labels(targets, PU_LABELS)
        is_positive = targets == Label.POSITIVE
        positive_risk = self.positive_prior * self.empirical_risk(
            scores[is_positive], targets[is_positive]
        )
        negative_risk = self.empirical_risk(
            scores[~is_positive], torch.full_like(targets[~is_positive], Label.NEGATIVE)
        ) - self.positive_prior * self.empirical_risk(
            scores[is_positive], torch.full_like(targets[is_positive], Label.NEGATIVE)
        )
        return positive_risk, negative_risk


class ImbalanceduPURiskEstimator(PositivePriorMixin, RiskEstimatorBase):
    def __init__(self, *args, expected_prior: float = 0.5, **kwargs) -> None:
        assert 0.0 < expected_prior < 1.0
        super().__init__(*args, **kwargs)
        self.expected_prior = expected_prior

    def forward(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert check_labels(targets, PU_LABELS)
        is_positive = targets == Label.POSITIVE
        positive_risk = self.expected_prior * self.empirical_risk(
            scores[is_positive], targets[is_positive]
        )
        negative_risk = ((1.0 - self.expected_prior) / (1.0 - self.positive_prior)) * (
            self.empirical_risk(
                scores[~is_positive],
                torch.full_like(targets[~is_positive], Label.NEGATIVE),
            )
            - self.positive_prior
            * self.empirical_risk(
                scores[is_positive],
                torch.full_like(targets[is_positive], Label.NEGATIVE),
            )
        )
        return positive_risk, negative_risk
