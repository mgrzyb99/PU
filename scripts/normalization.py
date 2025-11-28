import os
from contextlib import redirect_stdout

import torch
from torchvision.datasets import CIFAR10, KMNIST, MNIST, FashionMNIST
from torchvision.transforms import v2

ROOT = ".data/"
TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

if __name__ == "__main__":
    for dataset_cls in [MNIST, FashionMNIST, KMNIST, CIFAR10]:
        with open(os.devnull, "w") as f, redirect_stdout(f):
            dataset = dataset_cls(root=ROOT, train=True, transform=TRANSFORM)
        samples = torch.stack([sample for sample, _ in dataset])
        print(
            f"dataset: {dataset_cls.__name__}",
            f"μ = {torch.mean(samples, dim=[0, 2, 3])}",
            f"σ = {torch.std(samples, dim=[0, 2, 3])}",
            sep="\n",
        )
