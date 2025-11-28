from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


class MixupLossBase(nn.Module, ABC):
    def __init__(self, eps: float = 1e-4) -> None:
        assert eps > 0.0
        super().__init__()
        self.eps = eps

    @abstractmethod
    def forward(self, probas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass

    def clamp(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input, self.eps, 1.0 - self.eps)


class ChenMixupLoss(MixupLossBase):
    def forward(self, probas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert probas.shape == targets.shape
        probas, targets = self.clamp(probas), self.clamp(targets)
        return torch.mean((torch.log(probas) - torch.log(targets)) ** 2)


class ZhaoMixupLoss(MixupLossBase):
    def __init__(self, *args, kl: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kl = kl

    def forward(self, probas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert probas.shape == targets.shape
        probas, targets = self.clamp(probas), self.clamp(targets)
        loss = binary_cross_entropy(probas, targets)
        if self.kl:
            loss -= binary_cross_entropy(targets, targets)
        return loss
