import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from functools import reduce

def search_by_suffix(directory: str,
                     suffix: str) -> list:
    """find all the files with the suffix under the directory

    Args:
        directory (str): the directory to find the files
        suffix (str): the suffix of the files
    
    Returns:
        list: the list of the files
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                file_paths.append(os.path.join(root, file))
    return file_paths


def interpolate_weights(A: OrderedDict,
                        B: OrderedDict,
                        alpha: float,
                        beta: float,) -> OrderedDict:
    """interpolate the weights
    Args:
        A: the weights of model A
        B: the weights of model B
        alpha: the interpolation coefficient
        beta: the interpolation coefficient
    
    Returns:
        the interpolated weights
    """
    assert A.keys() == B.keys(), "the keys of A and B should be the same"
    C = OrderedDict()
    for k, v in A.items():
        if k.startswith("module."):
            k = k[7:]
        C[k] = alpha * v + beta * B[k]
    return C


def get_module(model: nn.Module, 
               module_name: str) -> nn.Module:
    """get the module from the model

    Args:
        model (nn.Module): the model to extract featuremaps.
        module_name (str): name of the module

    Returns:
        nn.Module: the module
    """
    return reduce(getattr, module_name.split('.'), model)


def evaluation(device: torch.device,
               model: nn.Module,
               testloader: DataLoader):
    """evaluate the model

    Args:
        device: GPU or CPU
        model: the model to evaluate
        testloader: the test dataset loader
    """
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # evaluatioin
    model.eval()
    with torch.no_grad():
        # testset
        test_losses, test_acc = [], 0
        for inputs, labels in tqdm(testloader):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the outputs
            outputs = model(inputs)
            # set the loss
            losses = loss_fn(outputs, labels)
            # set the loss and accuracy
            test_losses.extend(losses.cpu().detach().numpy())
            test_acc += (outputs.max(1)[1] == labels).sum().item()
    # print the test loss and accuracy
    test_loss = np.mean(test_losses)
    test_acc /= len(testloader.dataset)
    return test_loss, test_acc