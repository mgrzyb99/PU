import sys

from torchinfo import summary

sys.path.append("./")

from modules.model import KiryoCNN, KiryoMLP, TinyCNN

MNIST_SIZE = (30_000, 1, 28, 28)
CIFAR10_SIZE = (500, 3, 32, 32)

if __name__ == "__main__":
    summary(KiryoMLP(), input_size=MNIST_SIZE)
    summary(TinyCNN(), input_size=MNIST_SIZE)
    summary(KiryoCNN(), input_size=CIFAR10_SIZE)
    summary(TinyCNN(), input_size=CIFAR10_SIZE)
