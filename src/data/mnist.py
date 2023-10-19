import os, torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple

# import internal libs
from utils import get_logger

def load(root: str = "../data") -> Tuple[Dataset, Dataset]:
    """load the mnist dataset.
    Args:
        root (str): the root path of the dataset.
    Returns:
        return the dataset.
    """
    logger = get_logger(__name__)
    logger.info("loading mnist...")

    # prepare the transform
    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load the dataset
    os.makedirs(root, exist_ok=True)
    trainset = datasets.MNIST(
        root=root, train=True, download=True, transform=transform)
    testset = datasets.MNIST(
        root=root, train=False, download=True, transform=transform)

    # show basic info of dataset
    logger.info(f"trainset size: {len(trainset)}")
    logger.info(f"testset size: {len(testset)}")
    return trainset, testset