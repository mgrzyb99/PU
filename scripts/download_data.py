from torchvision.datasets import CIFAR10, KMNIST, MNIST, FashionMNIST

ROOT = ".data/"

if __name__ == "__main__":
    for dataset_cls in [MNIST, FashionMNIST, KMNIST, CIFAR10]:
        dataset_cls(root=ROOT, train=True, download=True)
        dataset_cls(root=ROOT, train=False, download=True)
