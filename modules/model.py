from torch import nn


class KiryoMLP(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Flatten(),
            nn.LazyLinear(300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )


class KiryoCNN(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.LazyConv2d(96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 10, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )


class TinyCNN(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.LazyConv2d(32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(1),
        )
